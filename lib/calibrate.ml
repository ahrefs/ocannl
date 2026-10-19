(** Memory-bandwidth calibration (gh-ocannl-578): STREAM-style streaming kernels whose byte counts
    are exact by construction — one access site per tensor node and direction, no guards — so the
    calibration rows they append under config [autotune_calibration_file] satisfy the fitter's
    per-leg exactness rule and can constrain the envelope's memory leg
    ([model_peak_memory_bandwidth]). Matmul-family tuning data cannot: its rows are compute-bound
    (their bytes-over-time understates achievable bandwidth no matter how exactly counted) and
    typically carry guards-taken/union upper bounds, so the fit of the memory leg needs exactly this
    kind of calibration diversity — see [tools/calibrate_bandwidth.ml] for the command-line entry
    point and [Ir.Cost_model.Calibration] for the fit semantics. *)

open Base
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
open Ocannl_tensor.Operation.DSL_modules

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(** [stream ?elems ?repeats ctx] times the four STREAM kernels — copy [c =: a], scale
    [b =: 0.5 *. a], add [c =: a + b], triad [a =: b + 0.5 *. c] — over 1-D single-precision tensors
    of [elems] cells each (default [2^26], 256 MiB per tensor: well past any last-level cache, so
    the measured rates are main-memory rates). Each kernel goes through {!Autotune.tune} with seed
    candidates only ([~rounds:0]) and the schedule cache disabled, so every timed candidate appends
    a calibration row through the ordinary emission path; the search is forced on ([~search:true]) —
    timing is the point of a calibration pass, whatever profile the config picked. Choose [elems] a
    power of two: an extent every workgroup size divides evenly keeps the parallelized candidates
    free of range guards, hence their rows bytes-exact.

    Returns the kernels' tuning reports in order, [(name, report)]. Raises {!Utils.User_error} when
    config [autotune_calibration_file] is not set — the pass would time kernels with nowhere to
    record them. *)
let stream ?(elems = 1 lsl 26) ?repeats ctx =
  let file =
    String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:"")
  in
  if String.is_empty file then
    raise
      (Utils.User_error
         "Calibrate.stream: config autotune_calibration_file is not set, so the calibration rows \
          would have nowhere to go");
  if elems <= 0 then raise (Utils.User_error "Calibrate.stream: elems must be positive");
  let a = NTDSL.term ~label:[ "stream_a" ] ~output_dims:[ elems ] () in
  let b = NTDSL.term ~label:[ "stream_b" ] ~output_dims:[ elems ] () in
  let c = NTDSL.term ~label:[ "stream_c" ] ~output_dims:[ elems ] () in
  List.iter [ a; b; c ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  (* The fills initialize the streams on device; values are irrelevant to the timings, and this
     routine is compiled outside [tune], so it contributes no calibration rows. *)
  let init =
    named "stream_init"
      [%cd
        a =: 2;
        b =: 0.5;
        c =: 0]
  in
  let init_ctx, routine = Train.to_routine ctx Idx.Empty init in
  Train.run init_ctx routine;
  let kernels =
    [
      ("stream_copy", [%cd c =: a]);
      ("stream_scale", [%cd b =: !.0.5 *. a]);
      ("stream_add", [%cd c =: a + b]);
      ("stream_triad", [%cd a =: b + (!.0.5 *. c)]);
    ]
  in
  (* Winner contexts are released as soon as their report is captured, and the initialization
     context — which owns the stream tensors' buffers — after the last kernel, exceptional paths
     included: a long-lived caller must not accumulate three large streams (plus winners) per
     calibration call. Leaves first, then their parent, per {!Context.release}'s precondition. *)
  Exn.protect
    ~finally:(fun () -> Context.release init_ctx)
    ~f:(fun () ->
      List.map kernels ~f:(fun (name, comp) ->
          let report = ref None in
          let tuned_ctx, (_routine : Context.routine) =
            (* [~name] as well as the block comment (gh-ocannl-669): the kernel's name is right
               here, so the rows' [routine] column is this string by construction, rather than by
               the coincidence that [get_name_exn]'s punctuation mangling leaves these names alone.
               The block comment stays — it is what labels the code in the generated artifacts. *)
            Autotune.tune ~name ~search:true ~rounds:0 ?repeats ~cache_dir:""
              ~report:(fun r -> report := Some r)
              init_ctx (named name comp) Idx.Empty
          in
          Context.release tuned_ctx;
          match !report with
          | Some r -> (name, r)
          | None ->
              (* [tune] reports exactly once per call on every path that does any work. *)
              raise
                (Utils.User_error (Printf.sprintf "Calibrate.stream: no tuning report for %s" name))))

(** {2 Cache-capacity calibration (gh-ocannl-1253)}

    The ranking score ({!Ir.Cost_model.ranking_seconds}) charges an untiled contraction's operand
    re-reads once its working set — per-block global read bytes times resident blocks — exceeds a
    capacity. That capacity is a fitted constant of the proxy, not the device's L2 size, so it is
    calibrated the way the bandwidth leg is: by a sweep whose kernels sit on both sides of it. *)

type ws_point = {
  wp_k : int;  (** The contraction's reduction extent. *)
  wp_loops : int list;
      (** The lowered loop extents of the nest writing the output, outermost first: [[k; m; n]] is
          backprop's weight-gradient order. *)
  wp_working_set : int;
      (** The contraction kernel's working set under the default plan, in bytes
          ({!Ir.Cost_model.reuse_traffic}). *)
  wp_requested : int;  (** Its global read bytes under per-block caching. *)
  wp_distinct : int;  (** Its distinct-cell bytes, read and written ({!Ir.Cost_model.analyze}). *)
  wp_ms : float;  (** The routine's best synced wall time over the repeats, in milliseconds. *)
  wp_rank_ms : float option;
      (** The kernel's ranking score under the capacity in force ({!Autotune.ranking_ms}). *)
  wp_bound_ms : float option;  (** Its distinct-cell roofline lower bound. *)
}

(** [working_set ?m ?n ?ks ?repeats ctx] times the untiled weight-gradient contraction
    [dw[o,i] = sum_k dy[k,o] * x[k,i]] — [m] rows of output (default 256), [n] columns (default
    1024: a gpt2_mini_train layer's [l*_ffn_w2] gradient) — at each reduction extent of [ks]
    (default [256] to [65536] in powers of two, batch 2 to 512 at sequence 128), on the plain
    default pipeline ({!Context.compile}: the plan whose re-reads the score charges, never a seeded
    or tuned one). The inputs are explicitly initialized host arrays (a standalone fixture: nothing
    upstream of the kernel is timed or left lazily initialized); the time is the whole routine's,
    synced, best of [repeats] (default 10) after one warm-up run. Each point carries the contraction
    kernel's working set and byte charges as the cost model reads them off the scheduled code. *)
let working_set ?(m = 256) ?(n = 1024)
    ?(ks = [ 256; 512; 1024; 2048; 4096; 8192; 16384; 32768; 65536 ]) ?(repeats = 10) ctx =
  if m <= 0 || n <= 0 || repeats <= 0 || List.exists ks ~f:(fun k -> k <= 0) then
    raise (Utils.User_error "Calibrate.working_set: extents and repeats must be positive");
  let backend_name = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let _capacity, max_resident_threads = Autotune.ranking_capacity ~limits in
  let cyc len modulus stride =
    Array.init len ~f:(fun c -> Float.of_int (1 + (c % modulus)) *. stride)
  in
  List.map ks ~f:(fun k ->
      let tag = Printf.sprintf "ws_k%d" k in
      (* A forward-ordered contraction: its output's loops outermost, the reduction innermost
         ([wp_loops] records it), where backprop lowers a weight gradient with the reduction loops
         outside. The sweep is self-consistent either way -- the bytes it charges are read off the
         very kernel it times -- and the fitted constants predicted the backprop-ordered
         [bench_wgrad] kernel within 13% on both HIP boxes (ahrefs/ocannl#1253). *)
      let dy = NTDSL.ndarray (cyc (k * m) 7 0.25) ~label:[ tag ^ "_dy" ] ~output_dims:[ k; m ] () in
      let x = NTDSL.ndarray (cyc (k * n) 5 0.125) ~label:[ tag ^ "_x" ] ~output_dims:[ k; n ] () in
      let%op dw = dy +* "ko; ki => oi" x in
      Train.set_materialized dw.Tensor.value;
      let segments = ref [] and loops = ref [] in
      let lowered_transform opt =
        (let rec writes = function
           | Ir.Low_level.For_loop { body; _ } | Ir.Low_level.If { body; _ } -> writes body
           | Ir.Low_level.Seq (a, b) -> writes a || writes b
           | Ir.Low_level.Set { tn; _ } -> Ir.Tnode.equal tn dw.Tensor.value
           | _ -> false
         in
         let rec nest acc = function
           | Ir.Low_level.For_loop { to_; body; _ } -> nest ((to_ + 1) :: acc) body
           | _ -> List.rev acc
         in
         List.iter (Ir.Low_level.flat_lines [ opt.Ir.Low_level.llc ]) ~f:(fun stmt ->
             if List.is_empty !loops && writes stmt then loops := nest [] stmt));
        let segs =
          Ir.Schedule.maybe_default_schedules ~backend_name ~limits ~static_indices:[] opt
        in
        segments := segs;
        segs
      in
      let comp = named (tag ^ "_wgrad") (Train.forward dw) in
      let run_ctx, routine =
        Context.compile ~name:(tag ^ "_wgrad") ~lowered_transform ctx comp Idx.Empty
      in
      Exn.protect
        ~finally:(fun () -> Context.release run_ctx)
        ~f:(fun () ->
          let reuse_of (o : Ir.Low_level.optimized) =
            Ir.Cost_model.reuse_traffic ?max_resident_threads ~is_global:(Ir.Cost_model.global_in o)
              o.Ir.Low_level.llc
          in
          let requested r = List.sum (module Int) r.Ir.Cost_model.ru_requested ~f:snd in
          (* The contraction's kernel: the segment requesting the most, beside its zero fill. *)
          let seg =
            List.max_elt !segments ~compare:(fun a b ->
                Int.compare (requested (reuse_of a)) (requested (reuse_of b)))
          in
          let r, distinct, scores =
            match seg with
            | Some o ->
                ( reuse_of o,
                  Ir.Cost_model.total_bytes (Ir.Cost_model.analyze o.Ir.Low_level.llc),
                  Autotune.ranking_ms ~limits [ o ] )
            | None -> ({ Ir.Cost_model.ru_requested = []; ru_working_set_bytes = 0 }, 0, None)
          in
          let ctx' = Context.run run_ctx routine in
          Context.sync ctx';
          let best = ref Float.infinity in
          for _ = 1 to repeats do
            let t0 = Unix.gettimeofday () in
            let ctx' = Context.run run_ctx routine in
            Context.sync ctx';
            best := Float.min !best ((Unix.gettimeofday () -. t0) *. 1000.)
          done;
          {
            wp_k = k;
            wp_loops = !loops;
            wp_working_set = r.Ir.Cost_model.ru_working_set_bytes;
            wp_requested = requested r;
            wp_distinct = distinct;
            wp_ms = !best;
            wp_rank_ms = Option.map scores ~f:fst;
            wp_bound_ms = Option.map scores ~f:snd;
          }))

type capacity_fit =
  | Transition of {
      cached_ws : int;  (** The largest working set still served at the cache rate. *)
      spilled_ws : int;  (** The smallest working set served at the DRAM rate. *)
      capacity : int;  (** Their geometric mean: the fitted [model_cache_capacity]. *)
      monotone : bool;
          (** Every cached point's working set is below every spilled point's. [false] means the
              sweep's rates do not split cleanly by working set, and the capacity is a guess. *)
      cache_rate : float;
          (** The median requested rate of the cached points, bytes/s: the fitted
              [model_cache_bandwidth]. *)
    }
  | All_cached of { largest_ws : int; cache_rate : float }
      (** Every point is served at the cache rate: the capacity is at least [largest_ws], and the
          sweep says no more than that (no capacity is invented; extend [ks] to find one). *)
  | All_spilled of { smallest_ws : int }
      (** Every point is served at the DRAM rate: the capacity is below [smallest_ws]. *)
  | No_transition of { contrast : float }
      (** Without a DRAM rate to compare against, the fastest and slowest requested rates differ by
          [contrast] < 2: the sweep cannot tell which side of the capacity it sits on. *)
  | Too_few_points

(** The capacity and cache bandwidth the sweep implies. A point's requested rate is its per-block
    read bytes over its time: within capacity, re-reads are served from cache and the rate sits far
    above DRAM bandwidth; past it, the rate falls to it. Points with a smaller working set than the
    fastest one are dropped first: a kernel too short to amortize its launch shows a LOW rate for a
    reason that is not the cache (tuf's 0.4 ms floor at the smallest extent read as spilled and made
    the first fit non-monotone). Over the rest, a point is cached when its rate is at least the
    geometric mean of the fastest rate and [dram_rate] (bytes/s, the calibrated DRAM bandwidth) —
    or, without one, of the fastest and the slowest rates, which needs a contrast of at least 2 to
    mean anything. The capacity is placed at the geometric mean of the working sets either side of
    the split — the model's decision boundary, between the shapes that demonstrably sit on each side
    of it — and the cache bandwidth is the cached points' median rate. *)
let fit_capacity ?dram_rate (points : ws_point list) : capacity_fit =
  let timed =
    List.filter points ~f:(fun p ->
        Float.(p.wp_ms > 0.)
        && Float.is_finite p.wp_ms && p.wp_requested > 0 && p.wp_working_set > 0)
  in
  (* Bytes per second. *)
  let rate p = Float.of_int p.wp_requested /. (p.wp_ms *. 1e-3) in
  let timed =
    match List.max_elt timed ~compare:(fun a b -> Float.compare (rate a) (rate b)) with
    | None -> []
    | Some fastest -> List.filter timed ~f:(fun p -> p.wp_working_set >= fastest.wp_working_set)
  in
  if List.length timed < 2 then Too_few_points
  else
    let rates = List.map timed ~f:rate in
    let hi = List.fold rates ~init:Float.neg_infinity ~f:Float.max in
    let lo = List.fold rates ~init:Float.infinity ~f:Float.min in
    let threshold =
      match dram_rate with
      | Some d -> Some (Float.sqrt (hi *. d))
      | None -> if Float.(hi /. lo < 2.) then None else Some (Float.sqrt (hi *. lo))
    in
    match threshold with
    | None -> No_transition { contrast = hi /. lo }
    | Some threshold -> (
        let cached, spilled = List.partition_tf timed ~f:(fun p -> Float.(rate p >= threshold)) in
        let ws p = p.wp_working_set in
        let median l =
          let sorted = List.sort (List.map l ~f:rate) ~compare:Float.compare in
          let n = List.length sorted in
          if n % 2 = 1 then List.nth_exn sorted (n / 2)
          else (List.nth_exn sorted ((n / 2) - 1) +. List.nth_exn sorted (n / 2)) /. 2.
        in
        let max_ws l = List.fold l ~init:0 ~f:(fun acc p -> max acc (ws p)) in
        let min_ws l = List.fold l ~init:Int.max_value ~f:(fun acc p -> min acc (ws p)) in
        match (cached, spilled) with
        | [], _ -> All_spilled { smallest_ws = min_ws spilled }
        | _, [] -> All_cached { largest_ws = max_ws cached; cache_rate = median cached }
        | _ ->
            let cached_ws = max_ws cached and spilled_ws = min_ws spilled in
            let lo_ws = min cached_ws spilled_ws and hi_ws = max cached_ws spilled_ws in
            Transition
              {
                cached_ws;
                spilled_ws;
                capacity = Float.to_int (Float.sqrt (Float.of_int lo_ws *. Float.of_int hi_ws));
                monotone = cached_ws < spilled_ws;
                cache_rate = median cached;
              })
