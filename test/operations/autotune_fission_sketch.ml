(* Autotune follow-ups: per-fission-segment tuning and the matmul sketch generator.

   Covered here, backend-independent (all printed booleans hold on every backend):

   - [Schedule.fission_scheduled] (the exposed fission pipeline with caller-supplied per-segment
   schedules) splits the canonical two-nest chain with a forced-materialized intermediate into two
   [`Normal] segments; compiling them through the [?lowered_transform] seam executes correctly. - A
   hand-crafted fissioned cache entry (per-segment schedules keyed by pre-schedule segment digests)
   replays through [Autotune.tune]'s cache-hit path: the report says [fissioned], and the values are
   correct — exercising [F_saved] rebinding of per-segment saved schedules. - [Autotune.tune] on a
   fissionable computation searches whole-routine and fissioned candidates and returns correct
   values; the second call hits the cache when the first search had no contention refusals, and
   otherwise retries; its search reaches several candidates, measuring the ones the backend can
   dispatch and accounting for the rest in the decline census (gh-ocannl-543 — on GPU backends only
   the fissioned preset is measured). - The matmul sketch generator detects a 32x32 matmul and seeds
   tile-size instantiations of the register-blocktiling (GPU) / operand-packing (CPU) pipelines,
   plus the tensorized (tile-MMA) pipelines (unstaged and cooperatively staged [Tensorize] on
   backends with an mma capability; whole-triple and Grid-split register-tiled [Tile_mma] on the C
   backends); the tuned routine matches the serial twin, and the schedules round-trip through the
   saved form when a sketch wins. - Per-fission-segment sketches ([F_sketch]): on a fissionable
   chain whose consumer is a matmul, the matmul's [Zero_out] fissions into its own [`Zeros] segment,
   so the whole-routine sketches never fit the segment's (unzeroed) site — the per-segment seeds
   must apply instead ([fiss_sketch_candidates]), get timed ([fiss_sketch_timed]), and the tuned
   routine matches the serial twin. *)

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

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let approx a b = Float.(abs (a - b) < 1e-2)

(* Which candidates a search times is host-dependent (gh-ocannl-892): a timing window that is mostly
   host stalls is refused ([Autotune.admitted_timing_ms]) and grows [timings_contended] instead of
   [candidates_timed]. The cross-machine sweep runs test/operations in parallel, and processes
   sharing one GPU refuse whole searches this small: the 09-03 cuda run emptied the chain search
   below, whose only dispatchable candidate on a GPU backend is the fissioned preset
   (gh-ocannl-543), so [candidates_timed] was 0 with nothing wrong in the tuner. Every claim here
   about what a search timed or measured therefore admits the load's own evidence as its one
   alternative -- a search that refused nothing still has to satisfy all of them -- and the
   accounting goes to stderr so a red run names the numbers instead of leaving them to be
   re-derived. *)
let accounting stage (r : Autotune.report) =
  Stdio.eprintf
    "%s (not part of the golden): %d candidate(s) timed, %d timing(s) refused over %d distinct \
     candidate(s), %d candidate(s) failed, default %s\n"
    stage r.Autotune.candidates_timed r.Autotune.timings_contended r.Autotune.candidates_contended
    r.Autotune.candidates_failed
    (match r.Autotune.default_ms with
    | Some _ -> "measured"
    | None -> if r.Autotune.default_refused then "refused" else "not measured")

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

(* The dune sandbox persists across runs: stale entries written by an older binary (digest
   ingredients and the saved-schedule format evolve) would break miss-then-hit assertions. *)
let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let () =
  List.iter
    [ "autotune_cache_fission"; "autotune_cache_fission2"; "autotune_cache_sketch" ]
    ~f:clean_cache;
  (* === The fissionable chain: d = a + b (forced materialized), e = d *. d^T. The transposed read
     keeps the cross-nest edge misaligned, so the aligned cross-nest rule cannot merge the pair and
     the chain still fissions (a plain pointwise consumer would now stay one kernel). === *)
  let n = 16 in
  let av =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let bv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let expected_e =
    Array.init (n * n) ~f:(fun idx ->
        let i = idx / n and j = idx % n in
        let dv k l = av.((k * n) + l) +. bv.((k * n) + l) in
        dv i j *. dv j i)
  in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n; n ] () in
  let%op d = a + b in
  Train.set_materialized d.Tensor.value;
  let%op e = d *. (d ++ "ij=>ji") in
  let chain_comp = named "af_chain" (Train.forward e) in

  (* --- fission_scheduled + the plural transforms seam, directly --- *)
  let seg_kinds = ref [] in
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let preset o =
          if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits o
          else if is_cpu then Sched.default_cpu ~min_parallel:1 o
          else []
        in
        let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
        seg_kinds := List.map tuples ~f:(fun (kind, _, _, _) -> kind);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      ctx chain_comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_e = Context.get_values ctx e.Tensor.value in
  p "fission_scheduled splits the chain into two normal segments"
    (match !seg_kinds with [ `Normal; `Normal ] -> true | _ -> false);
  p_all2 "fissioned segments through lowered_transform compute correctly" got_e expected_e ~f:approx;

  (* --- a hand-crafted fissioned cache entry replays through tune's cache-hit path --- *)
  let cache_dir = "autotune_cache_fission" in
  let base_capture = ref None in
  let bctx = Context.auto () in
  let _bctx, _br =
    Context.compile
      ~lowered_transform:(fun opt ->
        base_capture := Some opt;
        [ opt ])
      bctx chain_comp Ir.Indexing.Empty
  in
  let base_opt = Option.value_exn ~here:[%here] !base_capture in
  let base_canon = SC.canonicalize base_opt in
  let segments_assoc = ref [] in
  let fctx = Context.auto () in
  let _fctx, _fr =
    Context.compile
      ~lowered_transform:(fun opt ->
        let preset o =
          if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits o
          else if is_cpu then Sched.default_cpu ~min_parallel:1 o
          else []
        in
        let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
        segments_assoc :=
          List.filter_map tuples ~f:(fun (kind, pre, sched, _post) ->
              match kind with
              | `Normal ->
                  (* Per-segment replay matching keys on the structural canon (see
                     Schedule_cache.canonicalize's [with_placements]). *)
                  let pre_canon = SC.canonicalize ~with_placements:false pre in
                  let saved, _reg = SC.to_saved (SC.base_registry pre_canon) sched in
                  Some (SC.digest pre_canon, saved)
              | _ -> None);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      fctx chain_comp Ir.Indexing.Empty
  in
  let blimits = Context.hardware_limits bctx in
  SC.store ~dir:cache_dir
    ~key:(SC.cache_key ~limits:blimits base_canon ~backend:(Context.backend_name bctx))
    {
      SC.version = SC.entry_version;
      backend = Context.backend_name bctx;
      numerics = SC.numerics_tag ();
      codegen = Some (SC.codegen_tag ~limits:blimits ());
      objective = Some (SC.objective_tag ());
      source_digest = SC.digest base_canon;
      saved = [];
      segments = Some !segments_assoc;
      finer_fission = None;
      best_ms = 0.;
      baseline_ms = 0.;
      default_ms = None;
      (* Also pre-gh-579: no stored tensorized best, so a replay of it reports none. *)
      mma_best_ms = None;
      default_fingerprint = None;
    };
  let hit_report = ref None in
  let hctx = Context.auto () in
  let hctx, hroutine =
    Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> hit_report := Some r)
      hctx chain_comp Ir.Indexing.Empty
  in
  let hctx = Context.run hctx hroutine in
  let got_hit = Context.get_values hctx e.Tensor.value in
  (match !hit_report with
  | Some r ->
      p "hand-crafted fissioned entry hits the cache" (replayed r);
      p "cache-hit replay is fissioned" r.Autotune.fissioned
  | None ->
      p "hand-crafted fissioned entry hits the cache" false;
      p "cache-hit replay is fissioned" false);
  p_all2 "fissioned cache-hit replay computes correctly" got_hit expected_e ~f:approx;

  (* --- tune end-to-end on the fissionable chain (fresh cache dir: search, then hit) --- *)
  let cache_dir2 = "autotune_cache_fission2" in
  let reports = ref [] in
  let tune_chain () =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:cache_dir2
        ~report:(fun r -> reports := r :: !reports)
        ctx chain_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx e.Tensor.value
  in
  let got_t1 = tune_chain () in
  let got_t2 = tune_chain () in
  let r2, r1 =
    match !reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two reports"
  in
  accounting "chain search" r1;
  accounting "chain second call" r2;
  p_all2 "tuned fissionable chain values correct" got_t1 expected_e ~f:approx;
  let chain_first_cacheable = r1.Autotune.timings_contended = 0 in
  (* The load's own evidence, for the claims below that a refused search cannot satisfy — scoped to
     the report whose absence it explains, and within it to the candidate whose refusal explains it
     (Codex P2 on PR #608, twice). A refusal in one call says nothing about what the other could
     measure; and a report-wide refusal count says nothing about WHICH digest went unmeasured, so it
     cannot separate a contention-refused reference from the gh-ocannl-552 regression of never
     proposing or attributing the default seed. [default_refused] is that fact, recorded where the
     refusal happens. *)
  let reference_measured (r : Autotune.report) =
    match r.Autotune.default_ms with
    | Some d -> Float.is_finite d
    | None -> r.Autotune.default_refused
  in
  p "chain tune replays exactly after a contention-free search"
    (completed r1 && Bool.equal (replayed r2) chain_first_cacheable);
  let chain_cache_committed =
    chain_first_cacheable || (completed r2 && r2.Autotune.timings_contended = 0)
  in
  (* gh-ocannl-552: the untuned-default reference is measured by the search — the config-thresholds
     fissioned seed is the first candidate that binds a hardware dimension on GPU, and on CPU it is
     timed or dedups against a timed twin — and persists through the cache entry, so a cache-hit
     report still answers "did tuning beat the default?". The default seed's own window can be the
     refused one, which leaves no measurement to be the reference — admitted only against the
     refusal count (gh-ocannl-892). *)
  p
    "the default reference is measured and survives the cache round-trip, or is contention-refused \
     (gh-ocannl-552)"
    (reference_measured r1 && reference_measured r2
    &&
    match r1.Autotune.default_ms with
    | Some d1 -> Float.(r1.Autotune.best_ms <= d1)
    | None -> true);
  (* Codex P2 on PR #279: the cache key covers neither the scheduling gates nor the preset
     thresholds, so a config change can redefine the default pipeline without missing the cache.
     Simulated by rewriting the stored entry's fingerprint: the entry still hits — the winner replay
     is config-independent — but the config-relative default reference is dropped. *)
  let key2 = SC.cache_key ~limits:blimits base_canon ~backend:(Context.backend_name bctx) in
  (match SC.lookup ~dir:cache_dir2 ~key:key2 with
  | Some entry ->
      SC.store ~dir:cache_dir2 ~key:key2
        { entry with SC.default_fingerprint = Some "a-different-config" };
      let r3 = ref None in
      let c3 = Context.auto () in
      let c3, rt3 =
        Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:cache_dir2
          ~report:(fun r -> r3 := Some r)
          c3 chain_comp Ir.Indexing.Empty
      in
      let (_ : Context.t) = Context.run c3 rt3 in
      p "a stored stale default fingerprint drops the reference but not the hit (gh-ocannl-552)"
        (match !r3 with
        | Some r -> replayed r && Option.is_none r.Autotune.default_ms
        | None -> false)
  | None ->
      p "a stored stale default fingerprint drops the reference but not the hit (gh-ocannl-552)"
        (not chain_cache_committed));
  (* gh-ocannl-543: [candidates_timed >= 2] is a cc-shaped assertion. This chain's candidate space
     is the same on every backend, but most of it is serial forms — the whole-routine presets dedup
     to the unscheduled base, and the beam's Split/Swap/Vectorize moves off that base cannot bind a
     hardware dimension — so a GPU backend times exactly one candidate (the fissioned preset) and
     refuses the rest under gh-ocannl-532, where cc times all of them at full single-core speed. The
     portable statement is over the population the census now covers: at least one candidate reached
     a timing window, and the search reached several, whether measured or refused as unparallelized.
     A refused candidate is a candidate the search reached, so it belongs in that second count — but
     as a DISTINCT candidate, which [timings_contended] is not: it counts windows, and a refused
     digest is dropped from [seen] so an equivalent seed can retry it, so one candidate refused
     twice would stand for two (Codex P2 on PR #608). [candidates_contended] is the distinct-digest
     count, over refusals no later seed managed to time, so the three terms below partition the
     candidates this search reached and no waiver is needed. Reaching a window at all is a
     per-window fact and does take the raw sum. *)
  let not_dispatched (r : Autotune.report) =
    List.sum
      (module Int)
      r.Autotune.declines
      ~f:(fun d ->
        match d.Autotune.key with
        | Ir.Schedule_outcome.Not_dispatched_key _ -> d.Autotune.count
        | _ -> 0)
  in
  p "chain tune dispatched a candidate to a timing window (timed, or refused by contention)"
    (r1.Autotune.candidates_timed + r1.Autotune.timings_contended >= 1);
  p "chain tune reached multiple candidates (timed, refused by contention, or unparallelized)"
    (r1.Autotune.candidates_timed + r1.Autotune.candidates_contended + not_dispatched r1 >= 2);
  p "chain tune: candidates refused as unparallelized exactly on GPU"
    (if is_gpu then not_dispatched r1 > 0 else not_dispatched r1 = 0);
  p_all2 "chain cache-hit values correct" got_t2 expected_e ~f:approx;

  (* === The matmul sketch: 32x32 times 32x32 === *)
  let m = 32 in
  let mav =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let%op mc0 = ma * mb in
  let serial_comp = named "af_mm_serial" (Train.forward mc0) in
  let sctx = Context.auto () in
  let mm_capture = ref None in
  let sctx, sroutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        mm_capture := Some opt;
        [ opt ])
      sctx serial_comp Ir.Indexing.Empty
  in
  let sctx = Context.run sctx sroutine in
  let got_serial = nonzero "af_mm_serial" (Context.get_values sctx mc0.Tensor.value) in
  (* Whether the search proposes a tensorized sketch for a site at all is the backend's call, and
     [sketch_seed_params] is the composition the search enumerates, so the tests below ask it the
     same question rather than a backend-name substring: a GPU backend seeds [sk_mma] sketches only
     where its descriptor advertises an mma format for the site's f32 triple (Metal's simdgroup,
     CUDA's tf32 arm), and HIP's rocWMMA advertises none for f32, so on HIP every MMA counter of an
     f32 site is zero by design. The claims that count MMA candidates are then vacuous there and say
     so through [skipped], the way the sibling tuner tests treat a site with no tensorized seed; the
     scalar seeding and the value parity stay pinned on every backend. *)
  let tensorized_seeded ctx opt =
    Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits:(Context.hardware_limits ctx) opt
    |> List.exists ~f:(fun q -> q.Autotune.sk_mma)
  in
  (* The skip's scope says WHY the seed is missing. HIP's rocWMMA has no f32-input shape, so on HIP
     an f32 site never seeds a tensorized sketch whatever the configuration (verified with
     [tf32_matmuls] on): a backend fact. CUDA seeds one through the tf32 arm, which the repository
     default [tf32_matmuls=false] turns off: the measurement environment's choice, not the backend's
     limit, so that skip aggregates as environment non-coverage rather than as CUDA universally
     lacking the coverage (the sibling companion test draws the same line). *)
  let mma_skip_aggregation =
    match backend_name with
    | "cuda" when not (Ir.Numerics.get ()).tf32_matmuls -> `Environment
    | _ -> `Backend
  in
  let mma_claim ~seeded name b =
    if seeded then p name b
    else (
      Stdio.eprintf "af: no tensorized seed for this site on %s\n" backend_name;
      skipped ~aggregation:mma_skip_aggregation ~backend:backend_name name)
  in
  let mm_mma_seeded = tensorized_seeded sctx (Option.value_exn ~here:[%here] !mm_capture) in
  let%op mc1 = ma * mb in
  let mm_comp = named "af_mm_tuned" (Train.forward mc1) in
  let cache_dir3 = "autotune_cache_sketch" in
  let mm_reports = ref [] in
  let tune_mm () =
    let ctx = Context.auto () in
    (* [timing_ctx]: candidates compile and time against a scratch lineage; only the winner is
       compiled from [ctx] (the caller's buffers are never touched by timing runs). *)
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:cache_dir3
        ~timing_ctx:(Context.auto ())
        ~report:(fun r -> mm_reports := r :: !mm_reports)
        ctx mm_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx mc1.Tensor.value
  in
  let got_mm1 = tune_mm () in
  let got_mm2 = tune_mm () in
  let mr2, mr1 =
    match !mm_reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two mm reports"
  in
  p "matmul sketch instantiations seeded" (mr1.Autotune.sketch_candidates > 0);
  (* [mma_candidates] records the decision after the seeded candidates reached candidate compile; it
     is stronger than reconstructing a per-dialect minimum from the aggregate sketch count. *)
  mma_claim ~seeded:mm_mma_seeded "tensorized (mma) sketch instantiations seeded"
    (mr1.Autotune.mma_candidates > 0);
  Stdio.eprintf "matmul MMA accounting (not part of the golden): %d total, %d fission-scoped\n"
    mr1.Autotune.mma_candidates mr1.Autotune.fiss_mma_candidates;
  mma_claim ~seeded:mm_mma_seeded
    "whole-routine MMA candidates stay out of the fission-scoped counter"
    (mr1.Autotune.fiss_mma_candidates < mr1.Autotune.mma_candidates);
  p_all2 "tuned matmul matches the serial twin" got_mm1 got_serial ~f:approx;
  p "matmul tune replays exactly after a contention-free search"
    (completed mr1 && Bool.equal (replayed mr2) (mr1.Autotune.timings_contended = 0));
  p_all2 "matmul cache-hit values match the serial twin" got_mm2 got_serial ~f:approx;

  (* === Per-fission-segment sketches (F_sketch): qd = qa + qb (forced materialized), then the
     matmul qe = qd * qc. The chain fissions; the matmul's [Zero_out] lands in its own [`Zeros]
     segment, so the matmul segment's site is unzeroed — [detect_matmul] must fire per segment and
     the sketch pipelines apply without the zero-expansion geometry. === *)
  let q = 32 in
  let qav =
    Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus:11 ~offset:0. ~stride:0.125)
  in
  let qbv =
    Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let qcv =
    Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
  in
  let qa = TDSL.ndarray qav ~label:[ "qa" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let qb = TDSL.ndarray qbv ~label:[ "qb" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let qc = TDSL.ndarray qcv ~label:[ "qc" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let%op qd0 = qa + qb in
  Train.set_materialized qd0.Tensor.value;
  let%op qe0 = qd0 * qc in
  let fs_serial_comp = named "af_fs_serial" (Train.forward qe0) in
  let qsctx = Context.auto () in
  let qslimits = Context.hardware_limits qsctx in
  let fs_mma_seeded = ref false in
  let qsctx, qsroutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        (* The per-segment oracle asks the question the tuner asks, of the segments the tuner seeds:
           [F_sketch] entries come from [sketch_seed_params] on each post-fission [`Normal]
           segment's pre-schedule form, under the same fission pipeline and presets. The whole
           routine is the wrong subject here -- a materialized producer can fail the whole-routine
           tensorized family's companion coverage while the isolated matmul segment seeds fine
           (Codex P2 on PR #658). An unfissioned routine falls back to the whole-routine question,
           as the tuner does. *)
        let preset o =
          if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits:qslimits o
          else if is_cpu then Sched.default_cpu ~min_parallel:1 o
          else []
        in
        let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits:qslimits tns else [] in
        (fs_mma_seeded :=
           match
             Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices:[]
               opt
           with
           | [] | [ _ ] -> tensorized_seeded qsctx opt
           | tuples ->
               List.exists tuples ~f:(fun (kind, pre, _, _) ->
                   match kind with
                   | `Normal -> tensorized_seeded qsctx pre
                   | `Zeros | `Solo -> false));
        [ opt ])
      qsctx fs_serial_comp Ir.Indexing.Empty
  in
  let qsctx = Context.run qsctx qsroutine in
  let got_fs_serial = Context.get_values qsctx qe0.Tensor.value in
  let fs_mma_seeded = !fs_mma_seeded in
  let%op qd1 = qa + qb in
  Train.set_materialized qd1.Tensor.value;
  let%op qe1 = qd1 * qc in
  let fs_comp = named "af_fs_tuned" (Train.forward qe1) in
  let fs_report = ref None in
  let fsctx = Context.auto () in
  let fsctx, fsroutine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> fs_report := Some r)
      fsctx fs_comp Ir.Indexing.Empty
  in
  let fsctx = Context.run fsctx fsroutine in
  let got_fs = Context.get_values fsctx qe1.Tensor.value in
  (match !fs_report with
  | Some r ->
      (* The per-segment counter says that fission seeding happened; [mma_candidates] says a
         tensorized member actually reached candidate compile. The MMA counter is fission-scoped: a
         whole-routine MMA candidate cannot satisfy this assertion. *)
      p "per-segment sketch candidates seeded" (r.Autotune.fiss_sketch_candidates > 0);
      mma_claim ~seeded:fs_mma_seeded "per-segment tensorized sketch candidates seeded"
        (r.Autotune.fiss_mma_candidates > 0);
      p "per-segment sketch candidates timed" (r.Autotune.fiss_sketch_timed > 0)
  | None ->
      p "per-segment sketch candidates seeded" false;
      p "per-segment tensorized sketch candidates seeded" false;
      p "per-segment sketch candidates timed" false);
  p_all2 "tuned fissioned matmul matches the serial twin" got_fs got_fs_serial ~f:approx;

  (* === Multi-site F_sketch enumeration: two matmul segments with different geometries in one
     fissioned chain. Each parameter set of each site must be proposed ALONE (the other segment
     falling back to its default preset), so the combined candidate count is [site_a + site_b] — no
     site's seed can mask another site's seeds from being timed. Zipped combos fail this: index
     pairing (n-th combo = every segment's n-th set) gives [max site_a site_b], and pinning the
     others to their first set adds masking through an invalid first seed (observed on cifar_conv,
     PR #174: the fc matmul's invalid packrest-grid seed masked the conv segments' row-block seed).
     Cross-segment combination is recovered by one extra composite candidate that applies each
     site's best-timed single simultaneously — counted in [fiss_sketch_timed] but not in the seeded
     [fiss_sketch_candidates]. === *)
  let q2 = 16 in
  let qc16v =
    Array.init (q2 * q)
      ~f:(Ll_test.cycle_flat ~dims:[| q2; q |] ~modulus:9 ~offset:(-4.) ~stride:0.25)
  in
  let qc16 = TDSL.ndarray qc16v ~label:[ "qc16" ] ~input_dims:[ q ] ~output_dims:[ q2 ] () in
  let tune_candidates tag comp y =
    let report = ref None in
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ctx (named tag comp) Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx y.Tensor.value in
    match !report with
    | Some r ->
        accounting tag r;
        (r, got)
    | None -> failwith "expected a multi-site report"
  in
  let%op qd2 = qa + qb in
  Train.set_materialized qd2.Tensor.value;
  let%op qe2 = qd2 * qc in
  let report_a, _ = tune_candidates "af_ms_a" (Train.forward qe2) qe2 in
  let%op qd3 = qa + qb in
  Train.set_materialized qd3.Tensor.value;
  let%op qf3 = qc16 * qd3 in
  let report_b, _ = tune_candidates "af_ms_b" (Train.forward qf3) qf3 in
  let%op qd4 = qa + qb in
  Train.set_materialized qd4.Tensor.value;
  let%op qe4 = qd4 * qc in
  Train.set_materialized qe4.Tensor.value;
  let%op qg4 = qc16 * qe4 in
  let report_ab, got_ms = tune_candidates "af_ms_ab" (Train.forward qg4) qg4 in
  let%op qd5 = qa + qb in
  Train.set_materialized qd5.Tensor.value;
  let%op qe5 = qd5 * qc in
  Train.set_materialized qe5.Tensor.value;
  let%op qg5 = qc16 * qe5 in
  let ms_serial_comp = named "af_ms_serial" (Train.forward qg5) in
  let msctx = Context.auto () in
  let msctx, msroutine =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) msctx ms_serial_comp Ir.Indexing.Empty
  in
  let msctx = Context.run msctx msroutine in
  let got_ms_serial = Context.get_values msctx qg5.Tensor.value in
  let cand_a = report_a.Autotune.fiss_sketch_candidates in
  let cand_b = report_b.Autotune.fiss_sketch_candidates in
  let cand_ab = report_ab.Autotune.fiss_sketch_candidates in
  let timed_ab = report_ab.Autotune.fiss_sketch_timed in
  let eligible = report_ab.Autotune.fiss_sketch_composite_eligible in
  Stdio.eprintf "multi-site composite (not part of the golden): eligible=%b timed=%b\n" eligible
    report_ab.Autotune.fiss_sketch_composite_timed;
  p "multi-site: both sites seed per-segment sketches" (cand_a > 1 && cand_b > 1);
  p "multi-site: unmasked singles combo count (a + b)" (cand_ab = cand_a + cand_b);
  p "multi-site: best-timed singles recombined into a composite candidate"
    (* Refused single windows still count as timed, but cannot staff a composite. The eligibility
       fact comes from usable singles for two distinct segments, never from a report-wide contention
       waiver. Every eligible composite must reach its own window. *)
    (((not eligible) || report_ab.Autotune.fiss_sketch_composite_timed)
    && if is_cpu then timed_ab = cand_ab + Bool.to_int eligible else timed_ab > 0);
  p_all2 "multi-site: tuned two-matmul chain matches the serial twin" got_ms got_ms_serial ~f:approx;

  (* --- timing_ctx on a different backend is rejected (Codex P2 on PR #109): candidates timed
     elsewhere do not predict the target device, and the winner would be cached under the target
     backend's key without ever having been timed on it. sync_cc vs multicore_cc are both always
     available. --- *)
  p "timing_ctx on a different backend rejected"
    (match
       Autotune.tune ~rounds:0 ~repeats:1 ~cache_dir:"" ~timing_ctx:(Context.cpu ~threads:4 ())
         (Context.cpu ()) mm_comp Ir.Indexing.Empty
     with
    | _ -> false
    | exception Invalid_argument msg -> String.is_substring msg ~substring:"same backend")
