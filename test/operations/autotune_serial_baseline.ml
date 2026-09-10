(* gh-ocannl-532: the autotuner never dispatches an unparallelized candidate on a GPU backend.

   A kernel that binds no hardware dimension runs the whole routine in one work-item — at scalar
   throughput, unbounded in cost (a few-GFLOP training step measured 6.9 s per run on Metal, hours
   on HIP), uninterruptible, and sharing the device with the display. It cannot win a search whose
   other candidates are parallel, so it is not run at all: not timed, and not eligible to win.

   Printed booleans, all backend-independent:

   - The tuner's base compile binds no hardware dimension. Supplying a [?lowered_transform] bypasses
   the default annotator, so the identity-transform baseline is the unscheduled serial form on every
   backend — the premise of the rule. - The default compile of that same code does bind one wherever
   automatic scheduling is active, so on GPU the baseline is strictly the serial twin of code the
   backend parallelizes for free. - [Autotune.tune] times the baseline exactly where it is not a
   single work-item: [baseline_ms] is finite on CPU backends when its timing window is usable (the
   serial form runs at full single-core speed and stays a legitimate competitor), while a
   contention-dominated CPU window is explicitly counted and refused; it is [infinity] on GPU ones.
   Where dispatch itself is refused, the refusal is recorded in the report's decline census under
   [Not_dispatched_key "baseline"] (gh-ocannl-543). - Either way the search returns a working
   routine whose winner carries a measurement. - The rule holds on the cache-replay path too: a
   planted entry naming the serial form as the winner is rejected and re-searched on GPU, and
   honoured on CPU. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means instead
   of combining flags — [not (replayed r)] in particular does NOT say a search ran. *)
let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let approx a b = Float.(abs (a - b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Comfortably above [cpu_schedule_min_parallel] (16384) so the default annotator parallelizes on
   CPU backends too, not only on GPU ones (where the threshold is [gpu_schedule_min_parallel] = 64).
   Compiled, never run: this half of the test is structural. *)
let side = 256
let n = 16

let mav =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)

let mbv =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:11 ~offset:(-4.) ~stride:1.)

let mm_expected =
  Array.init (n * n) ~f:(fun idx ->
      let i = idx / n and j = idx % n in
      let acc = ref 0. in
      for k = 0 to n - 1 do
        acc := !acc +. (mav.((i * n) + k) *. mbv.((k * n) + j))
      done;
      !acc)

let () =
  (* --- The structural premise: identity transform = the serial form; no transform = parallel
     --- *)
  let av =
    Array.init (side * side)
      ~f:(Ll_test.cycle_flat ~dims:[| side; side |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let bv =
    Array.init (side * side)
      ~f:(Ll_test.cycle_flat ~dims:[| side; side |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let a = TDSL.ndarray av ~label:[ "sb_a" ] ~output_dims:[ side; side ] () in
  let b = TDSL.ndarray bv ~label:[ "sb_b" ] ~output_dims:[ side; side ] () in
  let%op sb_sum = a + b in
  let comp = named "sb_sum" (Train.forward sb_sum) in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  let base = Option.value_exn ~here:[%here] !captured in
  (* Over the lowering's loops: an elementwise routine with no loops at all would bind nothing for
     the wrong reason. *)
  p_empty "the tuner's base lowering binds no hardware dimension" ~over:(LL.loop_bounds base.LL.llc)
    (LL.hardware_axes base.LL.llc);
  let default =
    Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices:[] base
  in
  p "the untuned default compile of the same code does"
    ((not (Sched.automatic_schedule_active ~backend_name:backend))
    || List.exists default ~f:(fun o -> not (List.is_empty (LL.hardware_axes o.LL.llc))));

  (* --- The search: the baseline is timed on CPU backends and skipped on GPU ones --- *)
  let ma = TDSL.ndarray mav ~label:[ "sb_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "sb_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let tune_comp = named "sb_matmul" (Train.forward mc) in
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    (* [cache_dir:""] disables the disk cache: this test asserts what a search does, not what a
       previous run left behind. *)
    Autotune.tune ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx tune_comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx mc.Tensor.value in
  let r = Option.value_exn ~here:[%here] !report in
  let is_gpu = Sched.backend_is_gpu backend in
  let baseline_measured = Float.is_finite r.Autotune.baseline_ms in
  let baseline_contention_refused =
    (not is_gpu) && (not baseline_measured) && r.Autotune.timings_contended > 0
  in
  p "the serial baseline is measured or contention-refused on CPU, and not dispatched on GPU"
    (if is_gpu then not baseline_measured else baseline_measured || baseline_contention_refused);
  (* gh-ocannl-543: the refusal is a decline like any other. Without a census entry a GPU search
     that refused most of its candidate space reports exactly what an empty candidate space reports,
     and the difference was only visible in the [autotune_log] stderr stream. *)
  p "the refusal is recorded in the decline census, on GPU backends only"
    (Bool.equal is_gpu
       (List.exists r.Autotune.declines ~f:(fun d ->
            match d.Autotune.key with
            | Ir.Schedule_outcome.Not_dispatched_key origin -> String.equal origin "baseline"
            | _ -> false)));
  (* Which candidates a search times is host-dependent (gh-ocannl-892): a window that is mostly host
     stalls is refused ([Autotune.admitted_timing_ms]) and grows [timings_contended] instead of
     [candidates_timed]; processes sharing one GPU refuse whole searches this small (the 08-31 and
     09-02 hip sweeps ran the directory in parallel). So a search that timed nothing passes only on
     the load's own evidence: a search whose every candidate failed compile or dispatch shows no
     refusals and still fails here. *)
  let accounting stage (r : Autotune.report) =
    Stdio.eprintf
      "%s (not part of the golden): %d candidate(s) timed, %d timing(s) refused, %d candidate(s) \
       failed, default %s\n"
      stage r.Autotune.candidates_timed r.Autotune.timings_contended r.Autotune.candidates_failed
      (match r.Autotune.default_ms with Some _ -> "measured" | None -> "not measured")
  in
  accounting "search" r;
  p "the search timed at least one candidate, or was refused its timings by contention"
    (r.Autotune.candidates_timed >= 1 || r.Autotune.timings_contended > 0);
  p "the winner carries a measurement exactly when a candidate was timed"
    (Bool.equal (Float.is_finite r.Autotune.best_ms) (r.Autotune.candidates_timed >= 1));
  (* gh-ocannl-552: [baseline_ms] cannot answer "did tuning beat what the user gets without tuning?"
     on GPU (it is [infinity] there), so the untuned default pipeline's own seed is the reference.
     Attributed by digest: on CPU backends the config thresholds may leave the code unparallelized,
     in which case the seed dedups against the timed serial baseline and inherits its measurement.
     The default seed's own window can be the refused one, which leaves no measurement to be the
     reference -- admitted only against the refusal count. *)
  p "the untuned default pipeline is measured as the reference, or refused by contention (gh-552)"
    (match r.Autotune.default_ms with
    | Some d -> Float.is_finite d && Float.(r.Autotune.best_ms <= d)
    (* The default seed's OWN refusal, not the search's refusal count: report-wide, the count cannot
       separate this from the gh-552 regression of never proposing or attributing the seed (Codex P2
       on PR #608). *)
    | None -> r.Autotune.default_refused);
  p_all2 "tuned routine values correct" got mm_expected ~f:approx;

  (* --- The same rule on the cache-replay path. A cache entry written before the rule can name the
     serial baseline as the winner: it was timed then, and it wins by default whenever every
     candidate fails to compile. Such an entry is an empty saved schedule, which replays as the
     identity — so honouring it would reintroduce the single-work-item dispatch permanently, without
     ever timing anything. Planted by hand here, since the tuner no longer produces one. On CPU
     backends an empty schedule is a legitimate winner and must still hit. --- *)
  let cache_dir = "autotune_cache_serial_baseline" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let canon = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        [ opt ])
      ctx tune_comp Ir.Indexing.Empty
  in
  let canon = Option.value_exn ~here:[%here] !canon in
  let limits = Context.hardware_limits ctx in
  SC.store ~dir:cache_dir
    ~key:(SC.cache_key ~limits canon ~backend)
    {
      SC.version = SC.entry_version;
      backend;
      numerics = SC.numerics_tag ();
      codegen = Some (SC.codegen_tag ~limits ());
      objective = Some (SC.objective_tag ());
      source_digest = SC.digest canon;
      saved = [];
      segments = None;
      finer_fission = None;
      best_ms = 1e-6;
      baseline_ms = 1e-6;
      (* A pre-gh-552 entry: written before [default_ms] existed. *)
      default_ms = None;
      (* Also pre-gh-579: no stored tensorized best, so a replay of it reports none. *)
      mma_best_ms = None;
      default_fingerprint = None;
    };
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir
      ~report:(fun r -> report := Some r)
      ctx tune_comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx mc.Tensor.value in
  let r = Option.value_exn ~here:[%here] !report in
  accounting "poisoned-cache search" r;
  p "a serial cache entry is rejected on GPU backends and honoured on CPU ones"
    (Bool.equal (replayed r) (not is_gpu));
  (* A replay times nothing and refuses nothing, so on GPU either counter is evidence of the
     re-search; a refused window is still a window the replay would not have opened. *)
  p "rejecting it re-searches rather than returning the serial routine"
    (if is_gpu then r.Autotune.candidates_timed >= 1 || r.Autotune.timings_contended > 0
     else r.Autotune.candidates_timed = 0 && r.Autotune.timings_contended = 0);
  p "a pre-gh-552 entry reports no default measurement; a re-search measures one or is refused"
    (if is_gpu then Option.is_some r.Autotune.default_ms || r.Autotune.default_refused
     else Option.is_none r.Autotune.default_ms);
  p_all2 "the routine from the poisoned-cache path computes correct values" got mm_expected
    ~f:approx
