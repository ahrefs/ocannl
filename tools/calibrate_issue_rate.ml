(* Issue-rate and launch-fill calibration pass (gh-ocannl-1442, gh-ocannl-1479).

   Two sweeps of the weight-gradient contraction on the configured backend: - the reduction-extent
   sweep of [Ocannl.Calibrate.issue_sweep] (CUDA and HIP only: it times the seeded sketch), whose
   slope of time over the kernel's RAW scalar operations is the pre-fill [model_issue_rate] -- a
   rate that absorbs its kernel's partial launch fill; - the launch sweep of
   [Ocannl.Calibrate.launch_sweep] (any GPU backend): pinned scalar blocktile geometries over output
   sizes from a handful of blocks to many, fitted by [Ocannl.Calibrate.fit_fill] for the threads per
   multiprocessor at which scalar issue saturates ([model_fill_threads]) together with the full-fill
   issue rate (the normalized [model_issue_rate]).

   Usage: dune exec tools/calibrate_issue_rate.exe -- [--m=N] [--n=N] [--ks=K1,K2,...] [--repeats=N]
   [--multiprocessors=P] [--sizes=S1,S2,...] [--no-launch] [--ocannl_* flags]

   [--multiprocessors] is required where the device reports none (a Metal GPU with no IORegistry
   [gpu-core-count], gh-ocannl-1638); elsewhere it overrides the device's count. Stdout is the two
   tables, then the fits under [#] comment lines -- the raw fit, the fill fit with its residual, the
   uncapped form's best residual beside it, and the k-sweep predicted under the fill fit -- and
   config-pasteable lines: [model_issue_rate=] (when the fill fit succeeds, the reduction-extent
   sweep's slope over its kernel's fill-normalized operations, or the launch sweep's lead rate where
   there is no such sweep; the raw slope otherwise), [model_fill_threads=] and
   [model_multiprocessors=] when it succeeds. Exit 1 when no rate fits. Calibrate on an otherwise
   idle device: the fits read times. Startup chatter goes to stderr. *)

open Base
module Cal = Ocannl.Calibrate

let usage () =
  Stdio.eprintf
    "usage: calibrate_issue_rate [--m=N] [--n=N] [--ks=K1,K2,...] [--repeats=N] \
     [--multiprocessors=P] [--sizes=S1,S2,...] [--no-launch] [--ocannl_* flags]\n";
  Stdlib.exit 2

let int_list s =
  match List.map (String.split s ~on:',') ~f:Int.of_string with
  | l when List.for_all l ~f:(fun k -> k > 0) && not (List.is_empty l) -> l
  | _ | (exception _) -> usage ()

let () =
  let args = List.tl_exn (Array.to_list (Sys.get_argv ())) in
  let m = ref None and n = ref None and ks = ref None and repeats = ref None in
  let multiprocessors = ref None and sizes = ref None and launch = ref true in
  List.iter args ~f:(fun a ->
      let int_flag prefix =
        Option.map (String.chop_prefix a ~prefix) ~f:(fun v ->
            match Int.of_string v with v when v > 0 -> v | _ | (exception _) -> usage ())
      in
      match
        ( int_flag "--m=",
          int_flag "--n=",
          int_flag "--repeats=",
          int_flag "--multiprocessors=",
          String.chop_prefix a ~prefix:"--ks=",
          String.chop_prefix a ~prefix:"--sizes=" )
      with
      | Some v, _, _, _, _, _ -> m := Some v
      | _, Some v, _, _, _, _ -> n := Some v
      | _, _, Some v, _, _, _ -> repeats := Some v
      | _, _, _, Some v, _, _ -> multiprocessors := Some v
      | _, _, _, _, Some l, _ -> ks := Some (int_list l)
      | _, _, _, _, _, Some l -> sizes := Some (int_list l)
      | None, None, None, None, None, None ->
          if String.equal a "--no-launch" then launch := false
          else if not (Utils.cmdline_arg_is_config_key a) then usage ());
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let ms = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.4f") in
  let points =
    if Autotune.seeded_default_backend backend then (
      let points = Cal.issue_sweep ?m:!m ?n:!n ?ks:!ks ?repeats:!repeats ctx in
      Stdio.printf "# backend %s: the seeded sketch, threshold 32, split and sink off\n" backend;
      Stdio.printf "# %8s %14s %10s %8s %10s %12s %10s %10s\n" "k" "scalar_ops" "mma_ops" "kernels"
        "ms" "Gops/s" "rank_ms" "bound_ms";
      List.iter points ~f:(fun p ->
          Stdio.printf "  %8d %14d %10d %8d %10.4f %12.2f %10s %10s\n" p.Cal.ip_k
            p.Cal.ip_scalar_ops p.Cal.ip_mma_ops p.Cal.ip_kernels p.Cal.ip_ms
            (Float.of_int p.Cal.ip_scalar_ops /. (p.Cal.ip_ms *. 1e-3) /. 1e9)
            (ms p.Cal.ip_rank_ms) (ms p.Cal.ip_bound_ms));
      points)
    else (
      Stdio.printf "# backend %s: no reduction-extent sweep (the seeded default is CUDA/HIP only)\n"
        backend;
      [])
  in
  let raw =
    match Cal.fit_issue_rate points with
    | Cal.Issue_fit { rate; latency_ms; max_residual; points } ->
        Stdio.printf
          "# raw fit over %d points: time = %.4f ms + ops / %.6e ops/s (largest relative residual \
           %.3f); the intercept is not priced\n"
          points latency_ms rate max_residual;
        Some rate
    | Cal.Issue_unfittable why ->
        if not (List.is_empty points) then Stdio.printf "# no raw issue rate: %s\n" why;
        None
  in
  let fill =
    if not !launch then None
    else
      match
        match !multiprocessors with
        | Some p -> Some p
        | None -> Autotune.ranking_multiprocessors ~limits
      with
      | None ->
          Stdio.printf
            "# no launch fit: the device reports no multiprocessor count; pass --multiprocessors=P\n";
          None
      | Some p -> (
          let lpoints = Cal.launch_sweep ?sizes:!sizes ?repeats:!repeats ctx in
          match Cal.fit_fill ~multiprocessors:p lpoints with
          | Cal.Fill_unfittable why ->
              Stdio.printf "# launch sweep (%d points): no fill fit: %s\n" (List.length lpoints) why;
              None
          | Cal.Fill_fit f ->
              let fill_of ctxs =
                let ops =
                  Float.of_int (List.sum (module Int) ctxs ~f:(fun c -> c.Ir.Cost_model.lc_ops))
                in
                ops /. Cal.effective_ops ~multiprocessors:p ~fill_threads:f.saturation ctxs
              in
              Stdio.printf "# launch sweep, %d multiprocessors\n" p;
              Stdio.printf "# %14s %6s %8s %14s %10s %8s %12s %12s\n" "geometry" "size" "k"
                "scalar_ops" "ms" "fill" "Gops/s" "pred_ms";
              List.iter lpoints ~f:(fun lp ->
                  let ops =
                    List.sum (module Int) lp.Cal.lp_contexts ~f:(fun c -> c.Ir.Cost_model.lc_ops)
                  in
                  let pred =
                    Cal.effective_ops ~multiprocessors:p ~fill_threads:f.saturation
                      lp.Cal.lp_contexts
                    /. f.rate *. 1e3
                  in
                  Stdio.printf "  %14s %6d %8d %14d %10.4f %8.4f %12.2f %12.4f\n" lp.Cal.lp_geometry
                    lp.Cal.lp_m lp.Cal.lp_k ops lp.Cal.lp_ms (fill_of lp.Cal.lp_contexts)
                    (Float.of_int ops /. (lp.Cal.lp_ms *. 1e-3) /. 1e9)
                    pred);
              Stdio.printf
                "# fill fit over %d points (%d below 0.9 fill, %d starved left out): t = %d \
                 threads per multiprocessor, time = effective ops / %.6e ops/s, the lead \
                 geometry's rate (largest relative residual %.3f)\n"
                f.points f.partial_points f.starved_points f.saturation f.rate f.max_residual;
              Option.iter f.raw_rate ~f:(fun r ->
                  Stdio.printf
                    "# the same points over raw operations: %.6e ops/s (largest relative residual \
                     %s): the launch-blind reading\n"
                    r
                    (Option.value_map f.raw_residual ~default:"n/a" ~f:(Printf.sprintf "%.3f")));
              Option.iter f.uncapped ~f:(fun (t, r) ->
                  Stdio.printf
                    "# the uncapped form's best: t = %d, largest relative residual %.3f (the \
                     capped form ships)\n"
                    t r);
              List.iter points ~f:(fun ip ->
                  if not (List.is_empty ip.Cal.ip_contexts) then
                    Stdio.printf "# k-sweep k=%d: measured %.4f ms, fill %.4f, predicted %.4f ms\n"
                      ip.Cal.ip_k ip.Cal.ip_ms (fill_of ip.Cal.ip_contexts)
                      (Cal.effective_ops ~multiprocessors:p ~fill_threads:f.saturation
                         ip.Cal.ip_contexts
                      /. f.rate *. 1e3));
              Some (f.rate, f.saturation, f.cores))
  in
  match (fill, raw) with
  | Some (lead_rate, saturation, cores), _ ->
      (* The issue rate stays the reduction-extent sweep's (its seeded kernel is what the score's
         rate prices), normalized by that kernel's fill under the fitted saturation; the launch
         sweep's own lead rate stands in only where there is no such sweep (Metal). *)
      let rate =
        match
          Cal.fit_issue_rate
            ~fill:{ Ir.Cost_model.fm_multiprocessors = cores; fm_fill_threads = saturation }
            points
        with
        | Cal.Issue_fit { rate; max_residual; points = n; _ } ->
            Stdio.printf
              "# the reduction-extent sweep normalized by its kernel's fill: %.6e ops/s over %d \
               points (largest relative residual %.3f)\n"
              rate n max_residual;
            rate
        | Cal.Issue_unfittable _ -> lead_rate
      in
      Stdio.printf "model_issue_rate=%.6e\nmodel_fill_threads=%d\nmodel_multiprocessors=%d\n" rate
        saturation cores
  | None, Some rate -> Stdio.printf "model_issue_rate=%.6e\n" rate
  | None, None ->
      Stdio.printf "# no issue rate\n";
      Stdlib.exit 1
