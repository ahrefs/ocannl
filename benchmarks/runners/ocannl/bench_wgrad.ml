(* gh-ocannl-1183 kernel-level A/B: the forward + backprop routine of one dense layer at a
   gpt2_mini_train layer's sizes, whose weight gradient dw[o,i] = sum_{b,s} dy[b,s,o] * x[b,s,i] is
   lowered as backprop lowers every weight gradient of the training step -- the contraction loops
   (b, s) outermost. It measures the kernel the batch-scaling collapse of gh-ocannl-1183 is made of,
   without a fixture or a model around it.

   Usage: bench_wgrad.exe [ocannl configuration flags], with the environment selecting the workload
   and the arm. BENCH_WGRAD_DIMS is "b,s,o,i" (default "256,128,256,1024", the l*_ffn_w2 gradient at
   batch 256). BENCH_TUNE=1 times the tuned schedule (Autotune.tune, no disk cache) instead of the
   untuned default, which goes through Bench_harness.compile_default (Autotune.untuned_compile) and
   so honours the model_default_schedule and gpu_default_sketch_min_reduction settings.
   BENCH_REPEATS is the number of timed runs, min taken (default 20). BENCH_WGRAD_REQUIRE_SEEDED=1
   compiles the untuned arm through Autotune.seeded_default and exits 1, timing nothing, when it
   seeds no segment or falls back to the default.

   Prints one [wgrad:] line -- the arm, the dims, the w.grad nest's lowered loop extents, the
   min-of-repeats synced time of the whole routine, and for a tuned arm the search's crowned label
   and counters -- then the shipped kernels' isolated times ([Bench_harness.time_shipped_segments]),
   where the [w: wg_w.grad] row is the weight gradient's. Inputs are small positive dyadics, so the
   gradient's checksum is exact whatever the schedule and must agree across arms.

   Before timing, [wgrad-model:] lines give the cost model's view (gh-ocannl-1253), per kernel of
   two plans compiled for the purpose: [plan=default], the plain default pipeline (untiled), and
   [plan=seeded], the seeded untuned default's ([Autotune.seeded_transform]; absent where it does
   not apply). Each kernel row names the global nodes it writes, as the shipped-kernel rows do, with
   its ranking score ([rank_ms], the score [Autotune.model_default] minimizes, under the capacity in
   force) and its distinct-cell roofline ([bound_ms]); a [capacity=] line names the capacity and
   residency. Predicted against measured for the weight gradient's row is the calibration evidence
   for [model_cache_capacity]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module H = Bench_harness

let () =
  let dims =
    Option.value (Stdlib.Sys.getenv_opt "BENCH_WGRAD_DIMS") ~default:"256,128,256,1024"
    |> String.split ~on:',' |> List.map ~f:Int.of_string
  in
  let nb, ns, no, ni =
    match dims with
    | [ b; s; o; i ] -> (b, s, o, i)
    | _ -> failwith "BENCH_WGRAD_DIMS must be b,s,o,i"
  in
  let tune = Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_TUNE") (Some "1") in
  let require_seeded =
    Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_WGRAD_REQUIRE_SEEDED") (Some "1")
  in
  let repeats =
    Option.value_map (Stdlib.Sys.getenv_opt "BENCH_REPEATS") ~default:20 ~f:Int.of_string
  in
  let cyc n m stride = Array.init n ~f:(fun c -> Float.of_int (1 + (c % m)) *. stride) in
  (* The layer as the training step has it: [y = w * x] with a weight of input [i] and output [o]
     over activations of batch [b] and sequence [s]; the loss [sum (y *. dy)] makes [dy] the
     upstream gradient, so backprop's [w.grad] nest is [for b, s, o, i: w.grad[o,i] += dy[b,s,o] *
     x[b,s,i]] -- the shape every weight gradient of gpt2_mini_train has. *)
  let w =
    Operation.init ~l:"wg_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
      ~f:(fun idx -> Float.of_int (1 + ((idx.(0) + (3 * idx.(1))) % 5)) *. 0.125)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    NTDSL.ndarray
      (cyc (nb * ns * ni) 5 0.125)
      ~label:[ "wg_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
  in
  let dy =
    NTDSL.ndarray
      (cyc (nb * ns * no) 7 0.25)
      ~label:[ "wg_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
  in
  let%op y = w * x in
  let%op loss = (y *. dy) ++ "... => 0" in
  let wgrad = (Option.value_exn ~here:[%here] w.Tensor.diff).Tensor.grad in
  Train.set_materialized wgrad;
  let comp = Train.grad_update loss in
  let ctx = Context.auto () in
  (* The lowered loop order of the [w.grad] accumulation nest: the point of the workload is that it
     is backprop's, contraction loops outermost. *)
  let loops = ref [] in
  let _ =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        let rec writes = function
          | Ir.Low_level.For_loop { body; _ } | Ir.Low_level.If { body; _ } -> writes body
          | Ir.Low_level.Seq (a, b) -> writes a || writes b
          | Ir.Low_level.Set { tn; _ } -> Ir.Tnode.equal tn wgrad
          | _ -> false
        in
        let rec nest acc = function
          | Ir.Low_level.For_loop { to_; body; _ } -> nest ((to_ + 1) :: acc) body
          | _ -> List.rev acc
        in
        List.iter (Ir.Low_level.flat_lines [ opt.Ir.Low_level.llc ]) ~f:(fun stmt ->
            if List.is_empty !loops && writes stmt then loops := nest [] stmt);
        [ opt ])
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp Ir.Indexing.Empty
  in
  let backend = Context.backend_name ctx in
  (* The cost model's predictions per kernel of the default and the seeded plans
     (gh-ocannl-1253). *)
  let limits = Context.hardware_limits ctx in
  let capacity, resident = Autotune.ranking_capacity ~limits in
  let opt_int = Option.value_map ~default:"none" ~f:Int.to_string in
  let rate = Option.value_map ~default:"none" ~f:(Printf.sprintf "%.6g") in
  Stdio.printf
    "wgrad-model: capacity=%s resident_threads=%s cache_bandwidth=%s issue_rate=%s dram_bandwidth=%s\n"
    (opt_int capacity) (opt_int resident)
    (rate (Autotune.ranking_cache_bandwidth ~limits))
    (rate (Autotune.ranking_issue_rate ~limits))
    (rate (Autotune.ranking_dram_bandwidth ~limits));
  let written (o : Ir.Low_level.optimized) =
    List.filter_map (Ir.Cost_model.analyze o.Ir.Low_level.llc).Ir.Cost_model.per_node
      ~f:(fun (tn, fp) ->
        if fp.Ir.Cost_model.fp_write_bytes > 0 && Ir.Cost_model.global_in o tn then
          Some (Ir.Tnode.debug_name tn)
        else None)
    |> String.concat ~sep:","
  in
  let predict plan transform =
    match
      Context.compile_outcome ~lowered_transform:transform
        ~provenance:Ir.Schedule_outcome.User_schedule ctx comp Ir.Indexing.Empty
    with
    | Error _ -> Stdio.printf "wgrad-model: plan=%s did not compile\n" plan
    | Ok (plan_ctx, plan_routine) ->
        Context.release plan_ctx;
        (* The shipped kernels, after codegen's own rewrites of the list (GPU zero elision). *)
        List.iter plan_routine.Context.segments ~f:(fun o ->
            match Autotune.ranking_ms ~limits [ o ] with
            | Some (rank, bound) ->
                (* gh-ocannl-1442: the legs, so a prediction says which one it is. *)
                let leg = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.4f") in
                let compute, memory =
                  Option.value (Autotune.ranking_legs_ms ~limits [ o ]) ~default:(None, None)
                in
                Stdio.printf
                  "wgrad-model: plan=%s kernel=[%s] rank_ms=%.4f bound_ms=%.4f compute_ms=%s \
                   memory_ms=%s\n"
                  plan (written o) rank bound (leg compute) (leg memory)
            | None -> Stdio.printf "wgrad-model: plan=%s kernel=[%s] no coverage\n" plan (written o))
  in
  predict "default"
    (Ir.Schedule.maybe_default_schedules ~backend_name:backend ~limits ~static_indices:[]);
  Option.iter (Autotune.seeded_transform ctx Ir.Indexing.Empty) ~f:(predict "seeded");
  let report = ref None in
  let t0 = Unix.gettimeofday () in
  let ctx, routine =
    if tune then
      Autotune.tune ~search:true ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ctx comp Ir.Indexing.Empty
    else if require_seeded then (
      (* The seeded arm of a calibration (benchmarks/gh1253_cells.sh): compile through the seeded
         default itself and refuse to time a fallback under the seeded name -- the seeding is
         advisory, and a declined or recompiled plan is the default's. *)
      let choice = ref None in
      let result =
        Autotune.seeded_default ~report:(fun c -> choice := Some c) ctx comp Ir.Indexing.Empty
      in
      (match !choice with
      | Some c when (not (List.is_empty c.Autotune.sd_seeded)) && not c.Autotune.sd_fell_back ->
          Stdio.printf "wgrad: seeded %d segment(s)\n%!" (List.length c.Autotune.sd_seeded)
      | _ ->
          Stdio.eprintf
            "wgrad: BENCH_WGRAD_REQUIRE_SEEDED=1 but the seeded default seeded nothing or fell \
             back to the default; refusing to time it as the seeded plan\n\
             %!";
          Stdlib.exit 1);
      result)
    else H.compile_default ctx comp Ir.Indexing.Empty
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let best = ref Float.infinity in
  for _ = 1 to repeats do
    let c0 = Unix.gettimeofday () in
    let ctx = Context.run ctx routine in
    Context.sync ctx;
    best := Float.min !best ((Unix.gettimeofday () -. c0) *. 1000.)
  done;
  let ctx = Context.run ctx routine in
  let checksum = Array.fold (Context.get_values ctx wgrad) ~init:0. ~f:( +. ) in
  let tuned =
    match !report with
    | None -> ""
    | Some r ->
        Printf.sprintf
          " crowned=%s best_ms=%.4f baseline_ms=%.4f sketch_candidates=%d \
           fiss_sketch_candidates=%d timed=%d"
          r.Autotune.best_label r.Autotune.best_ms r.Autotune.baseline_ms
          r.Autotune.sketch_candidates r.Autotune.fiss_sketch_candidates r.Autotune.candidates_timed
  in
  Stdio.printf
    "wgrad: backend=%s arm=%s dims=%d,%d,%d,%d loops=[%s] compile_s=%.2f ms=%.4f checksum=%h%s\n%!"
    backend
    (if tune then "tuned" else if require_seeded then "seeded" else "default")
    nb ns no ni
    (String.concat ~sep:"," (List.map !loops ~f:Int.to_string))
    compile_s !best checksum tuned;
  ignore
    (H.time_shipped_segments ~repeats ~ctx ~bindings:Ir.Indexing.Empty [ routine ]
      : (float, string) Result.t list)
