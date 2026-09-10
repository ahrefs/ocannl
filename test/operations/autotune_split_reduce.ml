(* Split-reduce autotune seeding, gh-ocannl-484 task 3: the deterministic two-pass split reduction
   ([Sched.Split_reduce], landed as tasks 1/2/4) becomes reachable by the tuner. Covered here,
   backend-independent (all printed booleans hold on every backend):

   - [Autotune.split_reduce_sites] detects reduction-dominated sites: an rmw accumulation whose
   target has few cells while a long serial reduction loop feeds it (the skinny matvec below — the
   same shape as conv bias/weight gradients and split-K GEMMs). Elementwise code and short
   reductions yield no sites; the gh-466 embedding-backward scatter yields a dynamic site. -
   [Autotune.tune] seeds the sites as [F_split] candidates (report [split_reduce_candidates]),
   compiles and times them ([split_reduce_timed]) — the prelude applies whole-routine before
   fission, so the two passes land in separate kernels — and the tuned routine's values match the
   untuned serial reference (approximately: the split reassociates the reduction). - A hand-crafted
   split-reduce cache entry — whole-routine prelude in [saved] alongside post-prelude per-segment
   [segments] — replays through tune's cache-hit path ([F_split_saved]: [of_saved] re-mints the
   partials node, the per-segment lookup and drift guard pass), with correct values. - Multi-site:
   two skinny matvec accumulations in one routine seed per-site singles plus one composite
   recombining each site's best-timed [num_blocks]. - gh-ocannl-537: a bias gradient — the
   conv-benchmark shape, accumulated channel loop innermost — is reached through the enabling [Swap]
   chain and its candidate is TIMED, not merely seeded. - gh-ocannl-541: sites rank by estimated
   segment cost (no integer-division ratio-0 artefact), the candidate-volume cap lives in [tune],
   and a site the cap evicts is recorded in the decline census instead of vanishing. *)

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

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let approx a b = Float.(abs (a - b) < 1e-3 *. (1. +. abs b))

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

(* The dune sandbox persists across runs: stale entries written by an older binary would break
   miss-then-hit assertions. *)
let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let preset opt =
  if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits:Ir.Backend_intf.no_hardware_limits opt
  else if is_cpu then Sched.default_cpu ~min_parallel:1 opt
  else []

let zero_sched _tns = []

(* Skinny matvec — the reduction-dominated shape: y[o] = Σ_k m[o,k] * v[k], few output cells, long
   reduction. Mixed magnitudes so reassociation is numerically observable but small. *)
let out_d = 8
let red_d = 128

let mav =
  Array.init (out_d * red_d) ~f:(fun x ->
      Float.sin (Float.of_int x) *. (10. **. Float.of_int (x % 3)))

let mvv =
  Array.init red_d ~f:(fun x -> Float.cos (Float.of_int x) *. (10. **. Float.of_int (x % 2)))

let make_matvec label =
  let m =
    TDSL.ndarray mav ~label:[ "sr_m" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ out_d ] ()
  in
  let v = TDSL.ndarray mvv ~label:[ "sr_v" ^ label ] ~output_dims:[ red_d ] () in
  let%op y = m * v in
  y

let capture_base comp =
  let capture = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !capture

let run_forward ~name (t : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx (named name (Train.forward t)) Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

(* === Leg 1: site detection. === *)

let () =
  Tensor.unsafe_reinitialize ();
  let y = make_matvec "det" in
  let base_opt = capture_base (named "asr_detect" (Train.forward y)) in
  let sites = Autotune.split_reduce_sites base_opt in
  p "skinny matvec yields one static split-reduce site"
    (match sites with
    | [ s ] -> (not s.Autotune.sr_dynamic) && s.Autotune.sr_red = red_d && s.Autotune.sr_out = out_d
    | _ -> false);

  (* Elementwise code has no rmw accumulation: no sites. *)
  let a = TDSL.ndarray mvv ~label:[ "sr_a" ] ~output_dims:[ red_d ] () in
  let b = TDSL.ndarray mvv ~label:[ "sr_b" ] ~output_dims:[ red_d ] () in
  let%op w = a + b in
  let ew_opt = capture_base (named "asr_elemwise" (Train.forward w)) in
  p_empty "elementwise code yields no split-reduce sites" ~over:(LL.loop_bounds ew_opt.LL.llc)
    (Autotune.split_reduce_sites ew_opt);

  (* A short reduction (below the extent threshold) is not worth a second pass: no sites. *)
  let short = 32 in
  let ms =
    TDSL.ndarray
      (Array.init (out_d * short) ~f:(fun x -> Float.of_int (x % 5)))
      ~label:[ "sr_ms" ] ~input_dims:[ short ] ~output_dims:[ out_d ] ()
  in
  let vs =
    TDSL.ndarray
      (Array.init short ~f:(fun x -> Float.of_int (x % 3)))
      ~label:[ "sr_vs" ] ~output_dims:[ short ] ()
  in
  let%op ys = ms * vs in
  let short_opt = capture_base (named "asr_short" (Train.forward ys)) in
  p_empty "short reductions yield no split-reduce sites" ~over:(LL.loop_bounds short_opt.LL.llc)
    (Autotune.split_reduce_sites short_opt)

(* === Leg 2: the gh-466 embedding-backward scatter yields a dynamic site. === *)

let () =
  let vocab = 5 in
  let embed = 4 in
  let positions = 96 in
  let classes = TDSL.range vocab in
  let ids =
    TDSL.ndarray
      (Array.init positions ~f:(fun i -> Float.of_int (i % vocab)))
      ~label:[ "asr_ids" ] ~batch_dims:[ positions ] ~output_dims:[] ()
  in
  let%op one_hot = classes = ids in
  let c =
    TDSL.param
      ~values:(Array.init (embed * vocab) ~f:Float.of_int)
      "asr_c" ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let%op embedded = c * one_hot in
  let%op loss = embedded ++ "...|... => |->0" in
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Train.IDX.empty loss in
  let capture = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        [ opt ])
      ctx (named "asr_emb" update) Ir.Indexing.Empty
  in
  let emb_opt = Option.value_exn ~here:[%here] !capture in
  let sites = Autotune.split_reduce_sites emb_opt in
  p "embedding backward yields a dynamic (scatter) split-reduce site"
    (List.exists sites ~f:(fun s -> s.Autotune.sr_dynamic))

(* === Leg 3: tune seeds the site; tuned values match the serial reference. === *)

let () =
  let y0 = make_matvec "ref" in
  let want = nonzero "asr_serial" (run_forward ~name:"asr_serial" y0) in
  let y1 = make_matvec "tuned" in
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_tuned" (Train.forward y1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y1.Tensor.value in
  (match !report with
  | Some r ->
      (* [2*num_blocks <= red] admits b ∈ {8, 32} of the CPU sweep, b ∈ {32} of the GPU sweep. *)
      p "split-reduce candidates seeded"
        (if is_cpu then r.Autotune.split_reduce_candidates = 2
         else r.Autotune.split_reduce_candidates >= 1);
      p "split-reduce candidates timed" (r.Autotune.split_reduce_timed >= 1)
  | None ->
      p "split-reduce candidates seeded" false;
      p "split-reduce candidates timed" false);
  p_all2 "tuned skinny matvec matches the serial reference" got want ~f:approx

(* === Leg 4: a hand-crafted split-reduce cache entry replays (the F_split_saved path). === *)

let () =
  let cache_dir = "autotune_cache_split" in
  clean_cache cache_dir;
  let y0 = make_matvec "hcref" in
  let want = nonzero "asr_hc_serial" (run_forward ~name:"asr_hc_serial" y0) in
  let y1 = make_matvec "hc" in
  let hc_comp = named "asr_hc" (Train.forward y1) in
  let base_opt = capture_base hc_comp in
  let base_canon = SC.canonicalize base_opt in
  let prelude_saved = ref [] in
  let segments_assoc = ref [] in
  let sctx = Context.auto () in
  let _sctx, _sroutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let sites = Autotune.split_reduce_sites opt in
        let s = List.hd_exn sites in
        let op, _, _, _ =
          Sched.split_reduce ~axis:s.Autotune.sr_axis ~target:s.Autotune.sr_target ~num_blocks:8
        in
        prelude_saved := fst (SC.to_saved (SC.base_registry (SC.canonicalize opt)) [ op ]);
        let opt' = Sched.apply [ op ] opt in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt' in
        segments_assoc :=
          List.filter_map tuples ~f:(fun (kind, pre, sched, _post) ->
              match kind with
              | `Normal ->
                  let pre_canon = SC.canonicalize ~with_placements:false pre in
                  Some (SC.digest pre_canon, fst (SC.to_saved (SC.base_registry pre_canon) sched))
              | _ -> None);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      sctx hc_comp Ir.Indexing.Empty
  in
  p "the hand-crafted entry has a prelude and per-segment schedules"
    ((not (List.is_empty !prelude_saved)) && List.length !segments_assoc >= 2);
  let slimits = Context.hardware_limits sctx in
  SC.store ~dir:cache_dir
    ~key:(SC.cache_key ~limits:slimits base_canon ~backend:backend_name)
    {
      SC.version = SC.entry_version;
      backend = backend_name;
      numerics = SC.numerics_tag ();
      codegen = Some (SC.codegen_tag ~limits:slimits ());
      objective = Some (SC.objective_tag ());
      source_digest = SC.digest base_canon;
      saved = !prelude_saved;
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
      hctx hc_comp Ir.Indexing.Empty
  in
  let hctx = Context.run hctx hroutine in
  let got = Context.get_values hctx y1.Tensor.value in
  (match !hit_report with
  | Some r ->
      p "split-reduce entry hits the cache" (replayed r);
      p "split-reduce cache-hit replay is fissioned" r.Autotune.fissioned
  | None ->
      p "split-reduce entry hits the cache" false;
      p "split-reduce cache-hit replay is fissioned" false);
  p_all2 "split-reduce cache-hit replay computes correctly" got want ~f:approx

(* === Leg 5: multi-site — per-site singles plus the recombined composite. === *)

let () =
  let red2 = 256 in
  let m2v =
    Array.init (out_d * red2) ~f:(fun x -> Float.cos (Float.of_int x) *. Float.of_int ((x % 4) + 1))
  in
  let v2v = Array.init red2 ~f:(fun x -> Float.sin (Float.of_int (3 * x))) in
  let make_pair label =
    let m1 =
      TDSL.ndarray mav ~label:[ "ms_m1" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ out_d ] ()
    in
    let v1 = TDSL.ndarray mvv ~label:[ "ms_v1" ^ label ] ~output_dims:[ red_d ] () in
    let m2 =
      TDSL.ndarray m2v ~label:[ "ms_m2" ^ label ] ~input_dims:[ red2 ] ~output_dims:[ out_d ] ()
    in
    let v2 = TDSL.ndarray v2v ~label:[ "ms_v2" ^ label ] ~output_dims:[ red2 ] () in
    let%op y1 = m1 * v1 in
    Train.set_materialized y1.Tensor.value;
    let%op y2 = m2 * v2 in
    Train.set_materialized y2.Tensor.value;
    let%op z = y1 + y2 in
    z
  in
  let z0 = make_pair "ref" in
  let want = nonzero "asr_ms_serial" (run_forward ~name:"asr_ms_serial" z0) in
  let z1 = make_pair "tuned" in
  let base_opt = capture_base (named "asr_ms_probe" (Train.forward (make_pair "probe"))) in
  p "both accumulations are detected as split-reduce sites"
    (List.length (Autotune.split_reduce_sites base_opt) = 2);
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_ms_tuned" (Train.forward z1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx z1.Tensor.value in
  (match !report with
  | Some r ->
      (* CPU sweep {8, 32, 128}: site 1 (red 128) admits {8, 32}, site 2 (red 256) admits {8, 32,
         128} — five singles; the composite recombines the two sites' best-timed singles. Every
         single must reach a timing window. With no contention the composite is exactly one extra
         timing; a contention-refused single cannot staff it (gh-ocannl-855). *)
      p "multi-site split-reduce singles seeded"
        (if is_cpu then r.Autotune.split_reduce_candidates = 5
         else r.Autotune.split_reduce_candidates >= 2);
      p "best-timed singles recombined into a composite"
        (if is_cpu then
           (r.Autotune.split_reduce_timed - if r.Autotune.split_reduce_composite_timed then 1 else 0)
           = r.Autotune.split_reduce_candidates
           && Bool.equal r.Autotune.split_reduce_composite_timed
                r.Autotune.split_reduce_composite_eligible
         else r.Autotune.split_reduce_timed > 0)
  | None ->
      p "multi-site split-reduce singles seeded" false;
      p "best-timed singles recombined into a composite" false);
  p_all2 "tuned multi-site routine matches the serial reference" got want ~f:approx

(* === Leg 6: the conv-gradient shape — the enabling interchange (gh-ocannl-537). ===

   A bias gradient: the forward broadcast-adds [bias(o)] over [b,h,w,o], so the backward
   accumulation [bias.grad[o] += y.grad[b,h,w,o]] inherits the FORWARD product space — the
   accumulated channel loop INNERMOST, the reduction loops (b,h,w) outside it. That is how OCANNL
   lowers every parameter gradient of the conv benchmarks, and [Split_reduce] v1 rejects every axis
   of it: "the accumulation cell mentions <sym>, which is not bound by a loop enclosing the
   reduction loop". The family was therefore inert on the one segment it was filed for (89% of HIP
   lenet's step). Seeding now hoists the cell loop with a [Swap] chain and re-probes, so the site is
   detected WITH a non-empty interchange and — the assertion that matters, the
   seeded-but-never-timed failure mode of gh-476 — actually reaches timing. *)

let cg_b = 128
let cg_h = 4
let cg_w = 4
let cg_o = 6

let cgv =
  Array.init
    (cg_b * cg_h * cg_w * cg_o)
    ~f:(fun x -> Float.sin (Float.of_int x) *. (10. **. Float.of_int (x % 3)))

let make_bias_grad label =
  let x = TDSL.ndarray cgv ~label:[ "cg_x" ^ label ] ~output_dims:[ cg_b; cg_h; cg_w; cg_o ] () in
  let bias =
    TDSL.param
      ~values:(Array.init cg_o ~f:(fun i -> Float.of_int i /. 10.))
      ("cg_bias" ^ label) ~output_dims:[ cg_o ] ()
  in
  let%op y = x + bias in
  let%op loss = y ++ "bhwo => 0" in
  (* In the benchmarks an SGD step consumes the parameter gradient; here nothing does, so pin it
     materialized — the accumulation must exist as an on-device rmw for the site to be a site. *)
  Train.set_materialized (Option.value_exn ~here:[%here] bias.Tensor.diff).grad;
  (bias, loss)

let cg_grad ctx bias = Context.get_values ctx (Option.value_exn ~here:[%here] bias.Tensor.diff).grad

(* The bias gradient's own site: the one whose target has [cg_o] cells (the loss reduction is a
   second, already-splittable site — its accumulation cell is empty). *)
let cg_site sites = List.filter sites ~f:(fun s -> s.Autotune.sr_out = cg_o)

let () =
  let _bias_p, loss_p = make_bias_grad "probe" in
  let pctx = Context.auto () in
  let pctx = Train.init_params pctx Train.IDX.empty loss_p in
  let capture = ref None in
  let _pctx, _proutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        [ opt ])
      pctx
      (named "asr_cg_probe" (Train.grad_update loss_p))
      Ir.Indexing.Empty
  in
  let base_opt = Option.value_exn ~here:[%here] !capture in
  let sites = Autotune.split_reduce_sites base_opt in
  p "the bias gradient yields a split-reduce site"
    (match cg_site sites with [ s ] -> s.Autotune.sr_red = cg_b | _ -> false);
  p "the site carries an enabling interchange"
    (match cg_site sites with [ s ] -> not (List.is_empty s.Autotune.sr_swaps) | _ -> false);

  let bias0, loss0 = make_bias_grad "ref" in
  let rctx = Context.auto () in
  let rctx = Train.init_params rctx Train.IDX.empty loss0 in
  let rctx, rroutine =
    Context.compile rctx (named "asr_cg_serial" (Train.grad_update loss0)) Ir.Indexing.Empty
  in
  let rctx = Context.run rctx rroutine in
  let want = nonzero "asr_cg_serial" (cg_grad rctx bias0) in

  let bias1, loss1 = make_bias_grad "tuned" in
  let report = ref None in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Train.IDX.empty loss1 in
  let ctx, routine =
    (* Queued timing's batching semantics have dedicated executed coverage in
       [autotune_timing_modes]. Keep this stateful gradient arm about split-reduce reachability:
       under the queued objective each candidate is launched thousands of times, and the shape it
       launches is a Grid loop of extent 6 nested under a serial loop of 128 -- so on Apple
       platforms, where [cc_parallel_grid] probes to [dispatch], one launch is 128 [dispatch_apply]
       fork/joins and the search as a whole is millions of them. At that rate libdispatch's own
       allocator traps ("BUG IN CLIENT OF LIBMALLOC: memory corruption of free block", inside
       [_dispatch_calloc_typed] under [dispatch_apply]), and it does so with no OCANNL code in the
       process at all: [benchmarks/runners/ocannl/dispatch_apply_stress.c] reproduces the identical
       signature from a plain C loop (gh-ocannl-870). So this pin is not a workaround for a defect
       of the tuner's or of split-reduce's -- it keeps a platform trap out of a test about
       split-reduce reachability. *)
    Autotune.tune ~timing:Autotune.Isolated ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_cg_tuned" (Train.grad_update loss1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = cg_grad ctx bias1 in
  (match !report with
  | Some r ->
      (* Two sites — the bias gradient (interchanged) and the loss reduction (splittable as lowered)
         — over the [2*b <= extent] slice of each sweep: 4 singles on CPU, 2 on GPU. *)
      p "conv-gradient split-reduce candidates seeded"
        (if is_cpu then r.Autotune.split_reduce_candidates = 4
         else r.Autotune.split_reduce_candidates >= 2);
      (* The gh-537 assertion: TIMED, not merely seeded — seeded-but-never-timed is exactly the
         failure mode gh-476 paid a sweep to discover. Every single must reach a timing window. With
         no contention, [candidates + 1] additionally pins the composite, which is only proposed
         when EACH site contributed a usable best-timed single. Under contention a refused single
         cannot staff that composite, but still counts as reaching the timing window. *)
      p "the interchanged bias-gradient candidate reaches timing"
        ((r.Autotune.split_reduce_timed - if r.Autotune.split_reduce_composite_timed then 1 else 0)
         = r.Autotune.split_reduce_candidates
        && Bool.equal r.Autotune.split_reduce_composite_timed
             r.Autotune.split_reduce_composite_eligible)
  | None ->
      p "conv-gradient split-reduce candidates seeded" false;
      p "the interchanged bias-gradient candidate reaches timing" false);
  p_all2 "tuned bias gradient matches the serial reference" got want ~f:approx

(* === Leg 7: cost ranking and the eviction census (gh-ocannl-541). ===

   Three sites in one routine, chosen so the estimated-segment-cost ranking exactly INVERTS the old
   [sr_red / sr_out] integer-division ranking: a large-output matvec (out 512, red 128 — old ratio
   0, the artefact that excluded every conv weight gradient), a skinny matvec (out 8, red 128 — old
   ratio 16), and a scalar total (out 1, red 512 — old ratio 512). Trip counts 65536 / 1024 / 512
   rank them large-first. And with [max_split_reduce_sites:1] only the top site is seeded while the
   two evicted sites land in the decline census under [Seed_evicted_key "split_reduce"] — the gh-541
   Metal-leg blind spot was exactly a previously-seeded site vanishing without a signal when the cap
   bound. *)

let rk_o = 512

let rk_mv =
  Array.init (rk_o * red_d) ~f:(fun x ->
      Float.cos (Float.of_int (2 * x)) *. Float.of_int ((x % 3) + 1))

let make_ranked label =
  let mb =
    TDSL.ndarray rk_mv ~label:[ "rk_mb" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ rk_o ] ()
  in
  let vb = TDSL.ndarray mvv ~label:[ "rk_vb" ^ label ] ~output_dims:[ red_d ] () in
  let ms =
    TDSL.ndarray mav ~label:[ "rk_ms" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ out_d ] ()
  in
  let vs = TDSL.ndarray mvv ~label:[ "rk_vs" ^ label ] ~output_dims:[ red_d ] () in
  let%op yb = mb * vb in
  Train.set_materialized yb.Tensor.value;
  let%op ys = ms * vs in
  Train.set_materialized ys.Tensor.value;
  (* The scalar total of the large intermediate: red 512 over out 1 — the old ranking's #1.
     Materialized so its accumulation exists as an on-device rmw site; the short (red 8) total of
     [ys] stays below [sr_red_min] and contributes no site. *)
  let%op tb = yb ++ "o => 0" in
  Train.set_materialized tb.Tensor.value;
  let%op z = tb + (ys ++ "o => 0") in
  z

let () =
  let base_opt = capture_base (named "asr_rk_probe" (Train.forward (make_ranked "probe"))) in
  let sites = Autotune.split_reduce_sites base_opt in
  p "sites rank by estimated segment cost, inverting the old ratio-0 order"
    (match sites with
    | [ s1; s2; s3 ] ->
        s1.Autotune.sr_out = rk_o
        && s1.Autotune.sr_cost = rk_o * red_d
        && s2.Autotune.sr_out = out_d
        && s2.Autotune.sr_cost = out_d * red_d
        && s3.Autotune.sr_out = 1 && s3.Autotune.sr_cost = rk_o
    | _ -> false);
  let z0 = make_ranked "ref" in
  let want = nonzero "asr_rk_serial" (run_forward ~name:"asr_rk_serial" z0) in
  let z1 = make_ranked "capped" in
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~max_split_reduce_sites:1
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_rk_capped" (Train.forward z1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx z1.Tensor.value in
  (match !report with
  | Some r ->
      (* Only the top-ranked (large-output) site is seeded: red 128 admits b ∈ {8, 32} of the CPU
         sweep, b ∈ {32} of the GPU sweep. *)
      p "only the top-cost site is seeded under the cap"
        (if is_cpu then r.Autotune.split_reduce_candidates = 2
         else r.Autotune.split_reduce_candidates >= 1);
      p "the two evicted sites appear in the decline census"
        (List.exists r.Autotune.declines ~f:(fun d ->
             match d.Autotune.key with
             | Ir.Schedule_outcome.Seed_evicted_key "split_reduce" -> d.Autotune.count = 2
             | _ -> false));
      p "the census still sums to candidates_failed"
        (r.Autotune.candidates_failed
        = List.sum (module Int) r.Autotune.declines ~f:(fun d -> d.Autotune.count))
  | None ->
      p "only the top-cost site is seeded under the cap" false;
      p "the two evicted sites appear in the decline census" false;
      p "the census still sums to candidates_failed" false);
  p_all2 "the capped tuned routine matches the serial reference" got want ~f:approx;
  Stdio.printf "\nDone.\n%!"
