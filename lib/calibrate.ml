open Base
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
open Ocannl_tensor.Operation.DSL_modules

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Milliseconds on the monotonic clock since [counter] started. [Unix.gettimeofday] reads the system
   time, which Windows ticks in whole milliseconds or coarser, so a microsecond kernel timed as 0 ms
   there (gh-ocannl-1411); this is the clock the autotuner's own timings read. *)
let elapsed_ms counter = Mtime.Span.to_float_ns (Mtime_clock.count counter) /. 1e6

let stream ?(elems = 1 lsl 26) ?repeats ctx =
  let file = Utils.registered_text ~arg_name:"autotune_calibration_file" in
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

type ws_point = {
  wp_k : int;
  wp_loops : int list;
  wp_working_set : int;
  wp_requested : int;
  wp_distinct : int;
  wp_ms : float;
  wp_rank_ms : float option;
  wp_bound_ms : float option;
}

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
      let loops = ref [] in
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
        Ir.Schedule.maybe_default_schedules ~backend_name ~limits ~static_indices:[] opt
      in
      let comp = named (tag ^ "_wgrad") (Train.forward dw) in
      let run_ctx, routine =
        Context.compile ~name:(tag ^ "_wgrad") ~lowered_transform ctx comp Idx.Empty
      in
      Exn.protect
        ~finally:(fun () -> Context.release run_ctx)
        ~f:(fun () ->
          (* No capacity: the per-block charge alone, without a spilling one-block nest's serial
             re-reads (gh-ocannl-1633), which depend on the capacity this sweep fits. *)
          let reuse_of (o : Ir.Low_level.optimized) =
            Ir.Cost_model.reuse_traffic ?max_resident_threads ~is_global:(Ir.Cost_model.global_in o)
              o.Ir.Low_level.llc
          in
          let requested r = List.sum (module Int) r.Ir.Cost_model.ru_requested ~f:snd in
          (* The contraction's kernel: the shipped segment requesting the most. Read off the
             routine, after codegen's own rewrites of the kernel list (GPU zero elision drops the
             output's zero fill and folds it into the first accumulation), so the bytes charged are
             the code timed. *)
          let seg =
            List.max_elt routine.Context.segments ~compare:(fun a b ->
                Int.compare (requested (reuse_of a)) (requested (reuse_of b)))
          in
          let r, distinct, scores =
            match seg with
            | Some o ->
                ( reuse_of o,
                  Ir.Cost_model.total_bytes (Ir.Cost_model.analyze o.Ir.Low_level.llc),
                  Autotune.ranking_ms ~capabilities:(Context.codegen_capabilities ctx) ~limits [ o ]
                )
            | None ->
                ( { Ir.Cost_model.ru_requested = []; ru_working_set_bytes = 0; ru_scratch = [] },
                  0,
                  None )
          in
          let ctx' = Context.run run_ctx routine in
          Context.sync ctx';
          let best = ref Float.infinity in
          for _ = 1 to repeats do
            let t0 = Mtime_clock.counter () in
            let ctx' = Context.run run_ctx routine in
            Context.sync ctx';
            best := Float.min !best (elapsed_ms t0)
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
      cached_ws : int;
      spilled_ws : int;
      capacity : int;
      monotone : bool;
      cache_rate : float;
    }
  | All_cached of { largest_ws : int; cache_rate : float }
  | All_spilled of { smallest_ws : int }
  | No_transition of { contrast : float }
  | Too_few_points

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

type issue_point = {
  ip_k : int;
  ip_scalar_ops : int;
  ip_mma_ops : int;
  ip_kernels : int;
  ip_ms : float;
  ip_rank_ms : float option;
  ip_bound_ms : float option;
  ip_contexts : Ir.Cost_model.launch_context list;
}

let issue_sweep ?(m = 512) ?(n = 512) ?(ks = [ 256; 512; 1024; 2048; 4096; 8192; 16384; 32768 ])
    ?(repeats = 10) ctx =
  if m <= 0 || n <= 0 || repeats <= 0 || List.exists ks ~f:(fun k -> k <= 0) then
    raise (Utils.User_error "Calibrate.issue_sweep: extents and repeats must be positive");
  let limits = Context.hardware_limits ctx in
  let cyc len modulus stride =
    Array.init len ~f:(fun c -> Float.of_int (1 + (c % modulus)) *. stride)
  in
  List.map ks ~f:(fun k ->
      let tag = Printf.sprintf "ir_k%d" k in
      let dy = NTDSL.ndarray (cyc (k * m) 7 0.25) ~label:[ tag ^ "_dy" ] ~output_dims:[ k; m ] () in
      let x = NTDSL.ndarray (cyc (k * n) 5 0.125) ~label:[ tag ^ "_x" ] ~output_dims:[ k; n ] () in
      let%op dw = dy +* "ko; ki => oi" x in
      Train.set_materialized dw.Tensor.value;
      let choice = ref None in
      (* The seeded sketch at threshold 32, with the split and the reduction sink off: the
         one-kernel register-blocktiled contraction whose issue rate the ranking score prices,
         whatever gate the configuration picks. *)
      let lowered_transform =
        match
          Autotune.seeded_transform ~min_reduction:32 ~min_serial:0 ~reduction_sink:false
            ~on_choice:(fun c -> choice := Some c)
            ctx Idx.Empty
        with
        | Some t -> t
        | None ->
            raise
              (Utils.User_error
                 (Printf.sprintf
                    "Calibrate.issue_sweep: the seeded default does not apply on backend %s (CUDA \
                     and HIP only, with automatic scheduling and fission on)"
                    (Context.backend_name ctx)))
      in
      let comp = named (tag ^ "_wgrad") (Train.forward dw) in
      let run_ctx, routine =
        Context.compile ~name:(tag ^ "_wgrad") ~lowered_transform ctx comp Idx.Empty
      in
      Exn.protect
        ~finally:(fun () -> Context.release run_ctx)
        ~f:(fun () ->
          (match !choice with
          | Some c when (not (List.is_empty c.Autotune.sd_seeded)) && not c.Autotune.sd_fell_back ->
              ()
          | _ ->
              raise
                (Utils.User_error
                   (Printf.sprintf
                      "Calibrate.issue_sweep: k=%d did not seed the contraction (or fell back to \
                       the default): refusing to time it as the seeded sketch"
                      k)));
          let summary (o : Ir.Low_level.optimized) = Ir.Cost_model.analyze o.Ir.Low_level.llc in
          (* The contraction's kernel: the shipped segment with the most operations. *)
          let seg =
            List.max_elt routine.Context.segments ~compare:(fun a b ->
                Int.compare (summary a).Ir.Cost_model.flops (summary b).Ir.Cost_model.flops)
          in
          let scalar, mma, scores, contexts =
            match seg with
            | Some o ->
                let s = summary o in
                let capabilities = Context.codegen_capabilities ctx in
                let mma =
                  Ir.Cost_model.accepted_tile_mma_flops
                    ~operand_accepts:capabilities.Ir.Backend_intf.mma_accepts o
                in
                ( s.Ir.Cost_model.flops - mma,
                  mma,
                  Autotune.ranking_ms ~capabilities ~limits [ o ],
                  Ir.Cost_model.launch_contexts o.Ir.Low_level.llc )
            | None -> (0, 0, None, [])
          in
          let ctx' = Context.run run_ctx routine in
          Context.sync ctx';
          let best = ref Float.infinity in
          for _ = 1 to repeats do
            let t0 = Mtime_clock.counter () in
            let ctx' = Context.run run_ctx routine in
            Context.sync ctx';
            best := Float.min !best (elapsed_ms t0)
          done;
          {
            ip_k = k;
            ip_scalar_ops = scalar;
            ip_mma_ops = mma;
            ip_kernels = List.length routine.Context.segments;
            ip_ms = !best;
            ip_rank_ms = Option.map scores ~f:fst;
            ip_bound_ms = Option.map scores ~f:snd;
            ip_contexts = contexts;
          }))

type issue_fit =
  | Issue_fit of { rate : float; latency_ms : float; max_residual : float; points : int }
  | Issue_unfittable of string

let fit_issue_rate ?fill (points : issue_point list) : issue_fit =
  let timed =
    List.filter points ~f:(fun p ->
        p.ip_scalar_ops > 0 && Float.(p.ip_ms > 0.) && Float.is_finite p.ip_ms)
  in
  (* gh-ocannl-1479: under a fill model, each point's operations over its kernel's scalar fill --
     the work the score prices at the issue rate -- so the slope is a full-fill rate. *)
  let work p =
    let ops = Float.of_int p.ip_scalar_ops in
    match fill with
    | Some fm when not (List.is_empty p.ip_contexts) ->
        ops /. Ir.Cost_model.launch_fill fm p.ip_contexts
    | _ -> ops
  in
  let xs = List.map timed ~f:work in
  let ys = List.map timed ~f:(fun p -> p.ip_ms) in
  let count = List.length timed in
  let distinct = List.dedup_and_sort xs ~compare:Float.compare in
  if count < 2 || List.length distinct < 2 then
    Issue_unfittable "fewer than two timed points with distinct operation counts"
  else
    let nf = Float.of_int count in
    let mean l = List.fold l ~init:0. ~f:( +. ) /. nf in
    let mx = mean xs and my = mean ys in
    let sxy = List.fold2_exn xs ys ~init:0. ~f:(fun acc x y -> acc +. ((x -. mx) *. (y -. my))) in
    let sxx = List.fold xs ~init:0. ~f:(fun acc x -> acc +. ((x -. mx) *. (x -. mx))) in
    let slope = sxy /. sxx in
    (* ms per operation *)
    if Float.(slope <= 0.) then Issue_unfittable "the time does not grow with the operation count"
    else
      let latency_ms = my -. (slope *. mx) in
      let max_residual =
        List.fold2_exn xs ys ~init:0. ~f:(fun acc x y ->
            Float.max acc (Float.abs (latency_ms +. (slope *. x) -. y) /. y))
      in
      Issue_fit { rate = 1e3 /. slope; latency_ms; max_residual; points = count }

(* {2 Launch-fill calibration (gh-ocannl-1479)} *)

type launch_point = {
  lp_geometry : string;
  lp_m : int;
  lp_n : int;
  lp_k : int;
  lp_contexts : Ir.Cost_model.launch_context list;
  lp_ms : float;
}

let default_launch_geometries = [ (32, 32, 8, 4, 4); (64, 64, 8, 4, 4) ]
let default_launch_sizes = [ 32; 64; 128; 256; 512; 1024 ]

let launch_sweep ?(geometries = default_launch_geometries) ?(sizes = default_launch_sizes)
    ?(target_ops = 1 lsl 31) ?(max_k = 1 lsl 20) ?(repeats = 10) ctx =
  if
    repeats <= 0 || target_ops <= 0 || max_k <= 0
    || List.exists sizes ~f:(fun s -> s <= 0)
    || List.exists geometries ~f:(fun (bm, bn, bk, tm, tn) ->
        bm <= 0 || bn <= 0 || bk <= 0 || tm <= 0 || tn <= 0)
  then
    raise
      (Utils.User_error "Calibrate.launch_sweep: extents, geometries and repeats must be positive");
  let cyc len modulus stride =
    Array.init len ~f:(fun c -> Float.of_int (1 + (c % modulus)) *. stride)
  in
  List.concat_map geometries ~f:(fun (bm, bn, bk, tm, tn) ->
      let geometry = Printf.sprintf "%dx%dx%d/%dx%d" bm bn bk tm tn in
      List.filter_map sizes ~f:(fun size ->
          (* A tile wider than the output pads it: not this geometry's launch. *)
          if size < bm || size < bn || size % bm <> 0 || size % bn <> 0 then None
          else
            (* The reduction extent that puts about [target_ops] operations in the kernel, so a
               nearly empty launch still runs well above the launch latency: a power of two, a
               multiple of the reduction tile, at most [max_k]. *)
            let k =
              let want = max 1 (target_ops / (2 * size * size)) in
              let p = ref bk in
              while !p * 2 <= want && !p * 2 <= max_k do
                p := !p * 2
              done;
              !p
            in
            let tag = Printf.sprintf "fl_%dx%d_s%d_k%d" bm bn size k in
            let dy =
              NTDSL.ndarray
                (cyc (k * size) 7 0.25)
                ~label:[ tag ^ "_dy" ]
                ~output_dims:[ k; size ] ()
            in
            let x =
              NTDSL.ndarray
                (cyc (k * size) 5 0.125)
                ~label:[ tag ^ "_x" ]
                ~output_dims:[ k; size ] ()
            in
            let%op dw = dy +* "ko; ki => oi" x in
            Train.set_materialized dw.Tensor.value;
            let lowered_transform =
              Autotune.pinned_sketch_transform
                ~select:(fun p ->
                  p.Autotune.sk_bm = bm && p.sk_bn = bn && p.sk_bk = bk && p.sk_tm = tm
                  && p.sk_tn = tn && not p.sk_batch_grid)
                ctx Idx.Empty
            in
            let comp = named (tag ^ "_wgrad") (Train.forward dw) in
            let run_ctx, routine =
              Context.compile ~name:(tag ^ "_wgrad") ~lowered_transform ctx comp Idx.Empty
            in
            Exn.protect
              ~finally:(fun () -> Context.release run_ctx)
              ~f:(fun () ->
                let contexts =
                  (* The shipped kernels' contexts together: the zero nest's, if any, has no
                     operations and is absent. *)
                  List.concat_map routine.Context.segments ~f:(fun o ->
                      Ir.Cost_model.launch_contexts o.Ir.Low_level.llc)
                in
                let ctx' = Context.run run_ctx routine in
                Context.sync ctx';
                let best = ref Float.infinity in
                for _ = 1 to repeats do
                  let t0 = Mtime_clock.counter () in
                  let ctx' = Context.run run_ctx routine in
                  Context.sync ctx';
                  best := Float.min !best (elapsed_ms t0)
                done;
                Some
                  {
                    lp_geometry = geometry;
                    lp_m = size;
                    lp_n = size;
                    lp_k = k;
                    lp_contexts = contexts;
                    lp_ms = !best;
                  })))

type fill_form = Capped | Uncapped

let form_fill form ~multiprocessors ~fill_threads ~blocks ~threads =
  match form with
  | Capped ->
      Ir.Cost_model.context_fill
        { Ir.Cost_model.fm_multiprocessors = multiprocessors; fm_fill_threads = fill_threads }
        ~blocks ~threads
  | Uncapped ->
      Float.min 1.
        (Float.of_int (max 1 blocks)
        *. Float.of_int (max 1 threads)
        /. (Float.of_int (max 1 multiprocessors) *. Float.of_int (max 1 fill_threads)))

let effective_ops ?(form = Capped) ~multiprocessors ~fill_threads
    (contexts : Ir.Cost_model.launch_context list) =
  List.fold contexts ~init:0. ~f:(fun acc (c : Ir.Cost_model.launch_context) ->
      if c.lc_tile_mma then acc
      else
        acc
        +. Float.of_int c.lc_ops
           /. form_fill form ~multiprocessors ~fill_threads ~blocks:c.lc_blocks
                ~threads:c.lc_threads)

let scalar_ops (contexts : Ir.Cost_model.launch_context list) =
  List.sum (module Int) contexts ~f:(fun c -> if c.lc_tile_mma then 0 else c.lc_ops)

(* The proportional fit [y = x * slope] minimizing the squared RELATIVE residuals: with [u = x / y],
   [slope = sum u / sum u^2]. Relative, because a launch sweep's times span orders of magnitude and
   an absolute fit would serve its emptiest launch; through the origin, because the sweep holds the
   operations fixed, so an intercept would trade against the rate rather than measure a latency (the
   score prices none). Returns the slope, the sum of squared relative residuals and the largest
   relative residual; [None] on empty input or a non-finite fit. *)
let relative_proportional_fit (pairs : (float * float) list) =
  let us = List.map pairs ~f:(fun (x, y) -> x /. y) in
  let su = List.fold us ~init:0. ~f:( +. )
  and suu = List.fold us ~init:0. ~f:(fun a u -> a +. (u *. u)) in
  if List.is_empty pairs || Float.(suu <= 0.) then None
  else
    let slope = su /. suu in
    let rel u = (slope *. u) -. 1. in
    let sse = List.fold us ~init:0. ~f:(fun a u -> a +. (rel u *. rel u)) in
    let max_residual = List.fold us ~init:0. ~f:(fun a u -> Float.max a (Float.abs (rel u))) in
    if Float.is_finite slope && Float.(slope > 0.) && Float.is_finite sse then
      Some (slope, sse, max_residual)
    else None

type fill_fit =
  | Fill_fit of {
      cores : int;
      saturation : int;
      rate : float;
      max_residual : float;
      partial_points : int;
      points : int;
      starved_points : int;
      raw_rate : float option;
      raw_residual : float option;
      uncapped : (int * float) option;
    }
  | Fill_unfittable of string

let default_fill_candidates =
  [ 16; 24; 32; 48; 64; 96; 128; 160; 192; 256; 320; 384; 512; 640; 768; 1024; 1536; 2048 ]

(* A point's achieved scalar throughput, operations per millisecond. *)
let throughput p = Float.of_int (scalar_ops p.lp_contexts) /. p.lp_ms

let fit_fill ?(candidates = default_fill_candidates) ?(starved = 0.125) ~multiprocessors
    (points : launch_point list) : fill_fit =
  let timed =
    List.filter points ~f:(fun p ->
        scalar_ops p.lp_contexts > 0 && Float.(p.lp_ms > 0.) && Float.is_finite p.lp_ms)
  in
  let mixed =
    List.exists points ~f:(fun p -> List.exists p.lp_contexts ~f:(fun c -> c.lc_tile_mma))
  in
  (* Per geometry: its rate is its own (a wider tile reuses more), the saturation is the
     device's. *)
  let series =
    List.sort_and_group timed ~compare:(fun a b -> String.compare a.lp_geometry b.lp_geometry)
  in
  (* Latency-starved points -- below [starved] of their series' best throughput, a few warps per
     multiprocessor with nothing to hide a reduction step's latency behind -- sit on a regime the
     fill does not model; like the capacity fit's launch-bound points, they are left out. *)
  let series =
    List.map series ~f:(fun g ->
        let best = List.fold g ~init:0. ~f:(fun m p -> Float.max m (throughput p)) in
        List.filter g ~f:(fun p -> Float.(throughput p >= starved *. best)))
  in
  let kept = List.concat series in
  if multiprocessors <= 0 then Fill_unfittable "the multiprocessor count must be positive"
  else if mixed then Fill_unfittable "a point carries tile-MMA work: the fill is a scalar fit"
  else if List.length kept < 3 then
    Fill_unfittable "fewer than three timed, unstarved points with operations"
  else
    (* For one candidate [t]: each series' proportional fit, the summed squared relative residuals,
       the largest relative residual, and the first geometry's rate (the sweep's lead geometry, the
       seeded sketch's). *)
    let fit_with ~x t =
      let fits =
        List.map series ~f:(fun g ->
            relative_proportional_fit (List.map g ~f:(fun p -> (x t p, p.lp_ms))))
      in
      if List.exists fits ~f:Option.is_none then None
      else
        let fits = List.filter_opt fits in
        let sse = List.fold fits ~init:0. ~f:(fun a (_, e, _) -> a +. e) in
        let worst = List.fold fits ~init:0. ~f:(fun a (_, _, r) -> Float.max a r) in
        let lead =
          (* The lead geometry's series: the first timed point's, in the caller's order (the series
             are sorted by name, so their order says nothing). *)
          let g = (List.hd_exn timed).lp_geometry in
          List.find_map (List.zip_exn series fits) ~f:(fun (grp, (slope, _, _)) ->
              match grp with p :: _ when String.equal p.lp_geometry g -> Some slope | _ -> None)
        in
        Option.map lead ~f:(fun slope -> (t, (slope, sse, worst)))
    in
    let effective form t p = effective_ops ~form ~multiprocessors ~fill_threads:t p.lp_contexts in
    let best form =
      List.filter_map candidates ~f:(fit_with ~x:(effective form))
      |> List.min_elt ~compare:(fun (_, (_, e1, _)) (_, (_, e2, _)) -> Float.compare e1 e2)
    in
    match best Capped with
    | None -> Fill_unfittable "no candidate saturating thread count fits"
    | Some (t, (slope, _, max_residual)) ->
        let is_partial p =
          Float.(of_int (scalar_ops p.lp_contexts) < 0.9 *. effective Capped t p)
        in
        let partial = List.count kept ~f:is_partial in
        (* Identifiable only where some geometry's own points straddle the knee: with every point of
           each series partial (or every one full), changing [t] rescales each series' fill
           uniformly and its own rate absorbs it -- every candidate fits alike. *)
        let straddles =
          List.exists series ~f:(fun g ->
              List.exists g ~f:is_partial && List.exists g ~f:(fun p -> not (is_partial p)))
        in
        if partial < 2 then
          Fill_unfittable
            (Printf.sprintf
               "the sweep does not leave full fill under the best fit (%d threads per \
                multiprocessor): fewer than two points below 0.9 fill identify no saturation"
               t)
        else if not straddles then
          Fill_unfittable
            (Printf.sprintf
               "no geometry's points straddle the best fit's knee (%d threads per multiprocessor): \
                each series' own rate absorbs any saturation, which is unidentified"
               t)
        else
          let raw = fit_with ~x:(fun _ p -> Float.of_int (scalar_ops p.lp_contexts)) 0 in
          Fill_fit
            {
              cores = multiprocessors;
              saturation = t;
              rate = 1e3 /. slope;
              max_residual;
              partial_points = partial;
              points = List.length kept;
              starved_points = List.length timed - List.length kept;
              raw_rate = Option.map raw ~f:(fun (_, (sl, _, _)) -> 1e3 /. sl);
              raw_residual = Option.map raw ~f:(fun (_, (_, _, r)) -> r);
              uncapped = Option.map (best Uncapped) ~f:(fun (t, (_, _, r)) -> (t, r));
            }
