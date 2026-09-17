(* gh-ocannl-533, the containment half: a baseline the device cannot back is DECLINED, and the
   search survives it.

   [Autotune.tune] captures its base lowering by supplying a [?lowered_transform], which bypasses
   the default annotator — so the routine it links as the serial baseline is unscheduled,
   unfissioned, and unpromoted, and every [Local] intermediate of the whole computation sits in one
   work-item's stack frame. That is the worst case for private (scratch) memory, and on gfx1151 the
   [gpt2_mini] forward step asks for 163,856 bytes of it against a ~104 KB budget, so the
   gh-ocannl-533 pre-validator declines it at [Backend_link]. Until the base compile was protected
   like every other candidate's, that rejection propagated out of [tune] and killed the run before a
   single candidate had been tried — a correct diagnosis with the outcome the issue was filed about.

   The claim pinned here is the acceptance criterion: the over-budget baseline declines, the
   rejection is recorded in the decline census, the search continues, and the routine it returns
   computes the right values. What rescues the computation is fission plus the promotion of
   statement-crossing [Local]s to device memory — the same thing that lets [gpt2_mini] tune while
   every whole-routine preset declines — so the winner is a genuine result, not a survival artifact.

   The fixture is one deliberately oversized routine-local intermediate rather than a wide graph:
   [mid] is pinned [Local] and read back in an order that is not the write order, since a same-order
   read lets the compiler forward each store to its load and delete the array (the
   [hip_scratch_budget] lesson). Its size is just under the ~262 KB frame hipcc will emit, so the
   rejection is reachable on as many devices as possible: the budget is 4 GiB divided by the
   device's resident work-items, so it RISES on smaller devices, and one with <= 16384 of them
   cannot be pushed over budget by any compilable kernel.

   Gated behind [slow], and meaningful only where the baseline is actually over budget. The two ways
   to be inapplicable — a non-HIP backend, and a HIP device whose budget exceeds that frame — are
   announced on stderr rather than folded into the printed booleans, so the golden stays backend-
   and device-independent (the [hip_scratch_budget] idiom). The checks that do not depend on a
   decline (the search completes, times candidates, and computes the right value) are genuine on
   every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
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

(* 62500 floats = 250,000 B per work-item: over every scratch budget a device with more than 16384
   resident work-items reports, and ~12 KB under the frame hipcc refuses to emit. *)
let side = 250
let src_value i j = Float.of_int ((i + (3 * j)) % 5) *. 0.25
let expected_total = 2.0 *. (Float.of_int (side * side) *. 0.5)

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  if not (String.equal backend "hip") then
    Stdio.eprintf "scratch/tune: backend is %s, not hip — the decline checks below are vacuous\n%!"
      backend;

  let src =
    NTDSL.init ~l:"scrtune_src" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ side; side ]
      ~f:(fun idcs -> src_value idcs.(0) idcs.(1))
      ()
  in
  Train.set_materialized src.Tensor.value;
  let%op mid = src *. 2.0 in
  (* The whole point of the fixture: an intermediate the routine keeps in its own frame. *)
  Tn.update_memory_mode mid.Tensor.value Tn.Local (Site "991:test-setup");
  (* Read column-major while [mid] was written row-major, so the array survives to the frame. *)
  let%op col_sums = mid ++ "ij => j" in
  Train.set_materialized col_sums.Tensor.value;
  let%op total = col_sums ++ "j => 0" in
  let comp = Train.forward total in
  let report = ref None in
  (* [cache_dir:""] disables the disk cache: a hit reports no declines and runs no search, which
     here is indistinguishable from the bug being fixed. [rounds:0] keeps the search to its
     seeds. *)
  let ctx, routine =
    Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx comp Idx.Empty
  in
  let r = Option.value_exn ~here:[%here] !report in

  let scratch_declines =
    List.filter r.Autotune.declines ~f:(fun d ->
        match d.Autotune.key with SO.Resource_exceeded_key SO.Thread_scratch -> true | _ -> false)
  in
  if not r.Autotune.baseline_declined then
    Stdio.eprintf
      "scratch/tune: this device backs a %d-byte frame — the decline checks below are vacuous\n%!"
      (side * side * 4)
  else
    List.iter scratch_declines ~f:(fun d ->
        List.iter d.Autotune.sample_details ~f:(fun detail ->
            Stdio.eprintf "scratch/tune: declined: %s\n%!" detail));

  (* 1. The baseline was declined rather than fatal, and its rejection is in the census under the
     scratch key — the same aggregation any candidate's rejection gets. *)
  p "scratch/tune: the over-budget baseline is declined, not fatal"
    ((not r.Autotune.baseline_declined) || not (List.is_empty scratch_declines));
  p "scratch/tune: the declined baseline carries no measurement"
    ((not r.Autotune.baseline_declined) || Float.is_inf r.Autotune.baseline_ms);
  p "scratch/tune: the decline census sums to candidates_failed"
    (r.Autotune.candidates_failed
    = List.sum (module Int) r.Autotune.declines ~f:(fun d -> d.Autotune.count));
  (* One refusal, one census entry. gh-ocannl-543 records an unparallelized baseline as
     [Not_dispatched_key "baseline"], and a declined baseline is also never dispatched — but its
     reason is the rejection above, not "binds no hardware dimension", and counting it twice would
     inflate [candidates_failed] with a claim that is not true of this baseline. *)
  p "scratch/tune: the declined baseline is not also counted as a gh-532 refusal"
    ((not r.Autotune.baseline_declined)
    || not
         (List.exists r.Autotune.declines ~f:(fun d ->
              match d.Autotune.key with
              | SO.Not_dispatched_key origin -> String.equal origin "baseline"
              | _ -> false)));

  (* 2. The search continued to the next candidate: it completed, and timed something. This is the
     half that was broken while the diagnosis was already right. *)
  p "scratch/tune: the search completed" (completed r);
  p "scratch/tune: the search timed at least one candidate" (r.Autotune.candidates_timed >= 1);

  (* 3. And the winner is a real result: the tuned routine computes the value. *)
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx total.Tensor.value in
  p "scratch/tune: the tuned routine computes the right value"
    (Array.length got = 1 && Float.(abs (got.(0) - expected_total) < abs expected_total *. 1e-5));

  (* 4. The warm-cache shape, which is what the second run of the motivating workload does (Codex
     review, PR #271). The base compile happens BEFORE the cache lookup, so the baseline is declined
     on a cache hit too — and the report has to say so consistently: [baseline_declined] with an
     empty census would claim a rejection the census cannot account for. Nothing else can be in
     there, since no search ran. *)
  let cache_dir = "autotune_cache_hip_scratch" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let tune_cached () =
    let r = ref None in
    let ctx, routine =
      Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
        ~report:(fun rep -> r := Some rep)
        (Context.auto ()) comp Idx.Empty
    in
    let ctx = Context.run ctx routine in
    (Option.value_exn ~here:[%here] !r, Context.get_values ctx total.Tensor.value)
  in
  let populate, _ = tune_cached () in
  let hit, hit_got = tune_cached () in
  let hit_scratch_count =
    List.sum
      (module Int)
      (List.filter hit.Autotune.declines ~f:(fun d ->
           match d.Autotune.key with
           | SO.Resource_exceeded_key SO.Thread_scratch -> true
           | _ -> false))
      ~f:(fun d -> d.Autotune.count)
  in
  p "scratch/tune: the second run replays exactly after contention-free timing"
    (Bool.equal (replayed hit) (populate.Autotune.timings_contended = 0));
  p "scratch/tune: a cache hit still reports the declined baseline"
    (Bool.equal hit.Autotune.baseline_declined r.Autotune.baseline_declined);
  p "scratch/tune: the cache-hit census accounts for that decline, and only it"
    ((not hit.Autotune.baseline_declined)
    || (hit_scratch_count = 1 && hit.Autotune.candidates_failed = 1));
  p "scratch/tune: the cache-hit replay computes the right value"
    (Array.length hit_got = 1
    && Float.(abs (hit_got.(0) - expected_total) < abs expected_total *. 1e-5))
