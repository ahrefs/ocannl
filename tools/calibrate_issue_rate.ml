(* Issue-rate calibration pass (gh-ocannl-1442): sweep the seeded sketch of the weight-gradient
   contraction of [Ocannl.Calibrate.issue_sweep] over reduction extents on the configured backend
   (CUDA or HIP), and print the [model_issue_rate] its times imply -- the slope of time over the
   kernel's scalar operations, the effective rate the ranking score prices scalar work at.

   Usage: dune exec tools/calibrate_issue_rate.exe -- [--m=N] [--n=N] [--ks=K1,K2,...] [--repeats=N]
   [--ocannl_* flags]

   Stdout is a table, one row per extent -- the kernel's scalar and tile-MMA operations, the
   routine's kernels, its best time, the achieved rate, and the ranking score's prediction under the
   configuration in force (rank) beside the distinct-cell roofline (bound) -- then the fit under [#]
   comment lines (the intercept, which the score does not price, and the largest relative residual)
   and a config-pasteable [model_issue_rate=] line, or a [#] line saying why no line fits (exit 1).
   Calibrate on an otherwise idle device: the fit reads times. Startup chatter goes to stderr. *)

open Base
module Cal = Ocannl.Calibrate

let usage () =
  Stdio.eprintf
    "usage: calibrate_issue_rate [--m=N] [--n=N] [--ks=K1,K2,...] [--repeats=N] [--ocannl_* flags]\n";
  Stdlib.exit 2

let () =
  let args = List.tl_exn (Array.to_list (Sys.get_argv ())) in
  let m = ref None and n = ref None and ks = ref None and repeats = ref None in
  List.iter args ~f:(fun a ->
      let int_flag prefix = Option.map (String.chop_prefix a ~prefix) ~f:Int.of_string in
      match (int_flag "--m=", int_flag "--n=", int_flag "--repeats=") with
      | Some v, _, _ when v > 0 -> m := Some v
      | _, Some v, _ when v > 0 -> n := Some v
      | _, _, Some v when v > 0 -> repeats := Some v
      | Some _, _, _ | _, Some _, _ | _, _, Some _ -> usage ()
      | exception _ -> usage ()
      | None, None, None -> (
          match String.chop_prefix a ~prefix:"--ks=" with
          | Some list -> (
              match List.map (String.split list ~on:',') ~f:Int.of_string with
              | l when List.for_all l ~f:(fun k -> k > 0) && not (List.is_empty l) -> ks := Some l
              | _ | (exception _) -> usage ())
          | None -> if not (Utils.cmdline_arg_is_config_key a) then usage ()));
  let ctx = Context.auto () in
  let points = Cal.issue_sweep ?m:!m ?n:!n ?ks:!ks ?repeats:!repeats ctx in
  Stdio.printf "# backend %s: the seeded sketch, threshold 32, split and sink off\n"
    (Context.backend_name ctx);
  Stdio.printf "# %8s %14s %10s %8s %10s %12s %10s %10s\n" "k" "scalar_ops" "mma_ops" "kernels" "ms"
    "Gops/s" "rank_ms" "bound_ms";
  List.iter points ~f:(fun p ->
      let ms = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.4f") in
      Stdio.printf "  %8d %14d %10d %8d %10.4f %12.2f %10s %10s\n" p.Cal.ip_k p.Cal.ip_scalar_ops
        p.Cal.ip_mma_ops p.Cal.ip_kernels p.Cal.ip_ms
        (Float.of_int p.Cal.ip_scalar_ops /. (p.Cal.ip_ms *. 1e-3) /. 1e9)
        (ms p.Cal.ip_rank_ms) (ms p.Cal.ip_bound_ms));
  match Cal.fit_issue_rate points with
  | Cal.Issue_fit { rate; latency_ms; max_residual; points } ->
      Stdio.printf
        "# fitted over %d points: time = %.4f ms + ops / %.6e ops/s (largest relative residual \
         %.3f); the intercept is not priced\n"
        points latency_ms rate max_residual;
      Stdio.printf "model_issue_rate=%.6e\n" rate
  | Cal.Issue_unfittable why ->
      Stdio.printf "# no issue rate: %s\n" why;
      Stdlib.exit 1
