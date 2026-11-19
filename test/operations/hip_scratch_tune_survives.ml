(* gh-ocannl-533, the containment half: a candidate the device cannot back is DECLINED, and the
   search survives it.

   The HIP scratch validator ([Hip_backend.validate_scratch_budget]) declines at [Backend_link] a
   kernel whose private (scratch) segment the runtime could not back at full occupancy: 4 GiB
   divided by the device's resident work-items, about 104 KB on gfx1151. The decline is typed
   ([Resource_exceeded Thread_scratch]), so [Autotune.tune] must record it in the decline census and
   go on to the next candidate, and the routine it returns must compute the right values.

   This test was first written around the BASELINE: the base compile's unscheduled serial form held
   a whole routine's [Local]s in one work-item's frame, and the issue was that its decline killed
   the search. Since gh-ocannl-1340 a GPU baseline binds no hardware dimension and is refused before
   codegen, so it never meets the validator (and on a CPU backend there is no validator), which made
   every decline claim of the old form vacuous (gh-ocannl-1494). What remains reachable is a
   DISPATCHABLE candidate over budget, and nothing the search proposes for an ordinary computation
   is one: its fissioned candidates promote statement-crossing [Local]s to device memory (the [mid]
   below lands On_device), its whole-routine presets deduplicate to the refused baseline, and the
   compiler's own frame limit (gh-ocannl-1313) refuses a frame over ~261 KB before codegen. So the
   fixture builds the candidate: a wrapper over [Autotune.Context_driver] takes the FIRST searched
   candidate whose kernels bind a hardware dimension and prepends one kernel to it that keeps a
   [scratch_floats]-element [Local] array per Grid work-item (the [hip_scratch_budget] shape: filled
   forward, read back in reverse, so the compiler cannot forward the stores and drop the array). It
   is sized inside the window between the two limits -- over the scratch budget, under the frame
   limit -- so it passes every pre-codegen gate, gets compiled, and is declined at link. The
   prepended kernel rewrites [col_sums] with a value every candidate's later kernels overwrite, so
   even a candidate that linked would compute the same total.

   The candidate claims run in a leg open only where the decline is guaranteed by the device's own
   reported limits, never by the outcome: the HIP backend, [hip_scratch_validation] on, and the
   injected array's bytes over the budget computed from [max_resident_threads]. Elsewhere they are
   counted skips. The claims that never depended on a decline (the census arithmetic, the search
   completes, times candidates, computes the right value, replays from the cache) run on every
   backend, and the gh-ocannl-1340 fact that made the baseline claims vacuous is itself pinned on
   GPU backends: if a GPU baseline ever reaches codegen again, that claim fails and the
   baseline-decline contract needs its claims back. Gated behind [slow]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module B = Ll_builders
module Idx = Ir.Indexing
module SO = Ir.Schedule_outcome
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means instead
   of combining flags — [not (replayed r)] in particular does NOT say a search ran. *)
let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let side = 250
let src_value i j = Float.of_int ((i + (3 * j)) % 5) *. 0.25
let expected_total = 2.0 *. (Float.of_int (side * side) *. 0.5)

(* 62500 floats = 250,000 B per work-item: about 11 KB under the 261,112 B the HIP backend's frame
   gate leaves for declared arrays ([compiler_frame_limit_bytes - frame_reserve_bytes]), and over
   the scratch budget of every device with more than 17,176 resident work-items (gfx1151: 40,960, a
   104,832 B budget). *)
let scratch_floats = 62500
let scratch_bytes = scratch_floats * 4

(* The budget [Hip_backend.scratch_limit_per_work_item] computes, from the same documented constant
   (the 4 GiB WSL WDDM scratch cap) and the same queried residency. It decides only whether the leg
   below is open: a gate on the device's reported limits, not on the decline under test. *)
let scratch_budget ~resident =
  let granule = 64 and cap = 4 * 1024 * 1024 * 1024 in
  cap / resident / granule * granule

let scratch_tn =
  lazy
    (Tn.create ~namespace:"scrtune" (Tn.Specified Ir.Ops.single) ~id:0 ~label:[ "scratch"; "tune" ]
       ~unpadded_dims:(lazy [| scratch_floats |])
       ~padding:(lazy None)
       ())

(* One Grid work-item per column [j]: fill the scratch array with [src[0, j]] forward, then write it
   back to [col_sums[j]] in REVERSE, which leaves [col_sums[j] = src[0, j]] -- a value the
   candidate's own kernels then overwrite, since every one of them computes [col_sums] afresh. *)
let scratch_segment ~src ~col_sums (forms : LL.optimized list) : LL.optimized =
  let first = List.hd_exn forms in
  let scratch = Lazy.force scratch_tn in
  Tn.Placements.update first.LL.optimize_ctx.LL.placements scratch Tn.Local (Site "999:test-setup");
  let traced_store = Hashtbl.create (module Tn) in
  List.iter [ src; col_sums ] ~f:(fun tn ->
      match List.find_map forms ~f:(fun f -> Hashtbl.find f.LL.traced_store tn) with
      | Some traced -> Hashtbl.set traced_store ~key:tn ~data:traced
      | None -> ignore (LL.get_node traced_store tn : LL.traced_array));
  ignore (LL.get_node traced_store scratch : LL.traced_array);
  let j = B.sym () and x = B.sym () and y = B.sym () in
  let fill =
    B.loop_n x scratch_floats (B.set scratch [| B.iter x |] (B.get src [| B.fixed 0; B.iter j |]))
  in
  let drain =
    B.loop_n y scratch_floats
      (B.set col_sums [| B.iter j |] (B.get scratch [| B.aff [ (-1, y) ] (scratch_floats - 1) |]))
  in
  let llc = B.loop_n ~axis:LL.Grid j side (B.seq fill drain) in
  { first with LL.llc; traced_store }

let binds_hardware (opt : LL.optimized) = not (List.is_empty (LL.hardware_axes opt.LL.llc))

(* Armed before each search that should meet the over-budget candidate; [injected] records the label
   of the candidate that got the scratch kernel, and [injected_rejection] the typed phase and cause
   its compile came back with -- the census keeps the key and the detail but not the phase. *)
let inject : (Tn.t * Tn.t) option ref = ref None
let injected : string option ref = ref None
let injected_rejection : (SO.phase * SO.cause) option ref = ref None

module Injecting_driver = struct
  include Autotune.Context_driver

  let compile_outcome ?name ?lowered_transform ?prelowered ?reuse ~provenance ?candidate ?retain ctx
      comp bindings =
    let lowered_transform =
      match (lowered_transform, provenance, candidate, !inject, !injected) with
      | Some transform, SO.Candidate, Some c, Some (src, col_sums), None ->
          (* After the inner transform, so the search's in-transform gates (dedup, the gh-ocannl-532
             dispatchability refusal) decide first, as for any candidate. The gh-ocannl-1313 frame
             gate runs later, pre-driver in [Backends] on the segments this returns -- the injected
             kernel included -- which is why the array sits under the frame limit: only then does
             the candidate reach codegen and link. *)
          Some
            (fun opt ->
              let forms = transform opt in
              if List.exists forms ~f:binds_hardware then (
                injected := Some c;
                scratch_segment ~src ~col_sums forms :: forms)
              else forms)
      | transform, _, _, _, _ -> transform
    in
    let was_injected = Option.is_some !injected in
    let outcome =
      Autotune.Context_driver.compile_outcome ?name ?lowered_transform ?prelowered ?reuse
        ~provenance ?candidate ?retain ctx comp bindings
    in
    (if (not was_injected) && Option.is_some !injected then
       match outcome with
       | Error (SO.Classified { phase; cause; _ }) -> injected_rejection := Some (phase, cause)
       | Error (SO.Fatal _) -> ()
       | Ok _ ->
           (* The leg is open only where the device's limits guarantee the decline, so a link that
              succeeds is the validator regressing -- and returning the routine would have the
              search launch an over-budget kernel, which on WSL aborts the HSA queue rather than
              failing (gh-ocannl-533). Stop the search here, before any timing. *)
           failwith
             (Printf.sprintf
                "scratch/tune: the over-budget candidate %s LINKED: the HIP scratch validator \
                 accepted a %d-byte frame the device's limits put over budget"
                (Option.value !injected ~default:"?")
                scratch_bytes));
    outcome
end

module Injecting = Autotune.Make (Injecting_driver)

let scratch_count (r : Autotune.report) =
  List.sum
    (module Int)
    (List.filter r.Autotune.declines ~f:(fun d ->
         match d.Autotune.key with SO.Resource_exceeded_key SO.Thread_scratch -> true | _ -> false))
    ~f:(fun d -> d.Autotune.count)

let print_census tag (r : Autotune.report) =
  Stdio.eprintf
    "scratch/tune: %s census (not part of the golden): timed=%d failed=%d best=%s injected=%s\n%!"
    tag r.Autotune.candidates_timed r.Autotune.candidates_failed r.Autotune.best_label
    (Option.value !injected ~default:"(none)");
  List.iter r.Autotune.declines ~f:(fun d ->
      Stdio.eprintf "scratch/tune: %s census entry %s x%d: %s\n%!" tag
        (Sexp.to_string_hum (SO.sexp_of_rejection_key d.Autotune.key))
        d.Autotune.count
        (String.concat ~sep:" | " d.Autotune.sample_details))

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let on_hip = String.equal backend "hip" in
  let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts ctx) in
  let resident = (Context.hardware_limits ctx).Ir.Backend_intf.max_resident_threads in
  let validation = Utils.get_global_flag ~default:true ~arg_name:"hip_scratch_validation" in
  (* [None] where the device reports no usable residency, as the validator itself treats it. *)
  let budget =
    Option.bind resident ~f:(fun resident ->
        if resident <= 0 then None
        else Option.some_if (scratch_budget ~resident > 0) (scratch_budget ~resident))
  in
  if on_hip then
    Stdio.eprintf
      "scratch/tune: %d resident work-items, scratch budget %s B, injected array %d B, validation \
       %b (not part of the golden)\n\
       %!"
      (Option.value resident ~default:0)
      (Option.value_map budget ~default:"(none)" ~f:Int.to_string)
      scratch_bytes validation;
  let gpu_leg = Verdict.leg ~when_:on_gpu ~on:backend () in
  let hip_leg = Verdict.leg ~when_:on_hip ~on:backend () in
  let declines_by_limits =
    validation && Option.exists budget ~f:(fun budget -> scratch_bytes > budget)
  in
  let scratch_leg =
    Verdict.sub_leg ~aggregation:`Environment hip_leg
      ~when_:(lazy declines_by_limits)
      ~on:"a hip device whose scratch budget covers the fixture, or hip_scratch_validation=false"
  in
  let over_budget = on_hip && declines_by_limits in

  let src =
    NTDSL.init ~l:"scrtune_src" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ side; side ]
      ~f:(fun idcs -> src_value idcs.(0) idcs.(1))
      ()
  in
  Train.set_materialized src.Tensor.value;
  let%op mid = src *. 2.0 in
  (* A routine-local intermediate the fissioned candidates promote to device memory. *)
  Tn.update_memory_mode mid.Tensor.value Tn.Local (Site "991:test-setup");
  (* Read column-major while [mid] was written row-major, so the array survives to the frame. *)
  let%op col_sums = mid ++ "ij => j" in
  Train.set_materialized col_sums.Tensor.value;
  let%op total = col_sums ++ "j => 0" in
  let comp = Train.forward total in
  let arm () =
    injected := None;
    injected_rejection := None;
    inject := if over_budget then Some (src.Tensor.value, col_sums.Tensor.value) else None
  in
  let report = ref None in
  arm ();
  (* [cache_dir:""] disables the disk cache: a hit reports no declines and runs no search, which
     here is indistinguishable from the bug being fixed. [rounds:0] keeps the search to its
     seeds. *)
  let ctx, routine =
    Injecting.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx comp Idx.Empty
  in
  let r = Option.value_exn ~here:[%here] !report in
  let declined_label = !injected and declined_rejection = !injected_rejection in
  print_census "search" r;

  (* 1. Why the baseline is not the fixture: on a GPU backend it binds no hardware dimension and is
     refused before codegen (gh-ocannl-1340), once, under that cause — never declined at link. *)
  in_leg ~leg:gpu_leg "scratch/tune: the GPU baseline is refused before codegen, not declined"
    (fun label ->
      p label
        ((not r.Autotune.baseline_declined)
        && List.exists r.Autotune.declines ~f:(fun d ->
            SO.equal_rejection_key d.Autotune.key (SO.Not_dispatched_key "baseline")
            && d.Autotune.count = 1)));

  (* 2. The over-budget candidate was declined rather than fatal: its compile came back classified
     at [Backend_link] as [Resource_exceeded Thread_scratch] for at least the array's bytes, and the
     census holds it once under the scratch key, carrying the link validator's diagnosis
     (gh-ocannl-533) rather than the pre-codegen frame gate's; and it is not what the search
     returned. *)
  in_leg ~leg:scratch_leg
    "scratch/tune: the over-budget candidate is declined at link, censused under Thread_scratch"
    (fun label ->
      p label
        ((match declined_rejection with
           | Some
               ( SO.Backend_link,
                 SO.Resource_exceeded
                   { resource = SO.Thread_scratch; requested; limit = _; detail = _ } ) ->
               requested >= scratch_bytes
           | Some _ | None -> false)
        && scratch_count r = 1
        && List.exists r.Autotune.declines ~f:(fun d ->
            SO.equal_rejection_key d.Autotune.key (SO.Resource_exceeded_key SO.Thread_scratch)
            && List.exists d.Autotune.sample_details ~f:(fun detail ->
                String.is_substring detail ~substring:"gh-ocannl-533"))));
  in_leg ~leg:scratch_leg "scratch/tune: the declined candidate is not the winner" (fun label ->
      p label
        (match declined_label with
        | Some c -> not (String.equal c r.Autotune.best_label)
        | None -> false));
  p "scratch/tune: the decline census sums to candidates_failed"
    (r.Autotune.candidates_failed
    = List.sum (module Int) r.Autotune.declines ~f:(fun d -> d.Autotune.count));

  (* 3. The search continued past it: it completed, and timed something. This is the half that was
     broken while the diagnosis was already right. *)
  p "scratch/tune: the search completed" (completed r);
  p "scratch/tune: the search timed at least one candidate" (r.Autotune.candidates_timed >= 1);

  (* 4. And the winner is a real result: the tuned routine computes the value. *)
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx total.Tensor.value in
  p "scratch/tune: the tuned routine computes the right value"
    (Array.length got = 1 && Float.(abs (got.(0) - expected_total) < abs expected_total *. 1e-5));

  (* 5. The warm-cache shape, which is what the second run of the motivating workload does (Codex
     review, PR #271): the declined candidate must not be what the cache stores, so a replay links
     the winner alone and declines nothing — while a second run that searched again (its first run's
     timing was contended) meets the over-budget candidate again. *)
  let cache_dir = "autotune_cache_hip_scratch" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let tune_cached () =
    let r = ref None in
    arm ();
    let ctx, routine =
      Injecting.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
        ~report:(fun rep -> r := Some rep)
        (Context.auto ()) comp Idx.Empty
    in
    let ctx = Context.run ctx routine in
    (Option.value_exn ~here:[%here] !r, Context.get_values ctx total.Tensor.value)
  in
  let populate, _ = tune_cached () in
  let hit, hit_got = tune_cached () in
  print_census "cache hit" hit;
  let cache_claim =
    Test_utils.Cache_plant.unless_unkeyed ~identity:(Option.is_some (Context.timing_identity ctx))
  in
  cache_claim "scratch/tune: the second run replays exactly after contention-free timing"
    (Bool.equal (replayed hit) (populate.Autotune.timings_contended = 0));
  (* The populating search met the over-budget candidate too, so what it stored was chosen past a
     decline; the store writes the crowned, timed winner's form ([best_c.form] in [Autotune]), which
     a declined candidate can never be. *)
  in_leg ~leg:scratch_leg
    "scratch/tune: the populating search declined the candidate, its replay declines nothing"
    (fun label ->
      cache_claim label
        (scratch_count populate = 1 && Bool.equal (replayed hit) (scratch_count hit = 0)));
  p "scratch/tune: the cache-hit replay computes the right value"
    (Array.length hit_got = 1
    && Float.(abs (hit_got.(0) - expected_total) < abs expected_total *. 1e-5))
