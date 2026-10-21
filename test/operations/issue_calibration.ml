(* gh-ocannl-1442: the issue-rate calibration behind tools/calibrate_issue_rate.exe.

   [Calibrate.fit_issue_rate] fits [time = latency + ops / rate] by least squares over a sweep of
   the seeded sketch; its readings are exercised on synthetic sweeps -- an exact line, a launch
   floor that only moves the intercept, a sweep whose time does not grow, too few points -- and
   [Calibrate.issue_sweep] is run on the configured backend: at toy extents on CUDA and HIP, where
   only structural facts are claimed (every point seeded one scalar kernel whose operations grow
   with the extent), and elsewhere, where the seeded default does not apply and the sweep must
   refuse rather than time the default's kernel under the sketch's name. No time reaches the
   golden. *)

open Base
open Verdict.Claims
module Cal = Ocannl.Calibrate

let point ~ops ~ms =
  {
    Cal.ip_k = 1;
    ip_scalar_ops = ops;
    ip_mma_ops = 0;
    ip_kernels = 1;
    ip_ms = ms;
    ip_rank_ms = None;
    ip_bound_ms = None;
  }

let close a b = Float.(abs (a -. b) <= 1e-9 *. abs b)

let () =
  (* 2e12 operations/s with a 0.1 ms intercept: exact dyadic-free inputs, so compare closely. *)
  let line =
    List.map [ 1; 2; 4; 8 ] ~f:(fun g ->
        point ~ops:(g * 1_000_000_000) ~ms:(0.1 +. (0.5 *. Float.of_int g)))
  in
  p "an exact line fits its rate and intercept"
    (match Cal.fit_issue_rate line with
    | Cal.Issue_fit { rate; latency_ms; max_residual; points } ->
        close rate 2e12 && close latency_ms 0.1 && Float.(max_residual < 1e-9) && points = 4
    | Cal.Issue_unfittable _ -> false);
  p "points without operations are not fitted"
    (match Cal.fit_issue_rate (point ~ops:0 ~ms:5. :: line) with
    | Cal.Issue_fit { rate; points; _ } -> close rate 2e12 && points = 4
    | Cal.Issue_unfittable _ -> false);
  p "a time that does not grow with the operations fits nothing"
    (match Cal.fit_issue_rate [ point ~ops:1_000 ~ms:2.; point ~ops:2_000 ~ms:1. ] with
    | Cal.Issue_unfittable _ -> true
    | Cal.Issue_fit _ -> false);
  p "one operation count fits nothing"
    (match Cal.fit_issue_rate [ point ~ops:1_000 ~ms:1.; point ~ops:1_000 ~ms:1.1 ] with
    | Cal.Issue_unfittable _ -> true
    | Cal.Issue_fit _ -> false)

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let seeds_here = Autotune.seeded_default_backend backend_name

let () =
  let ks = [ 64; 256 ] in
  let sweep () = Cal.issue_sweep ~m:256 ~n:256 ~ks ~repeats:1 (Context.auto ()) in
  (* One golden on every backend: each claim is gated, and a closed gate prints the same row. *)
  let points = if seeds_here then sweep () else [] in
  gated_all ~min:(List.length ks) ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP every swept extent is timed, seeded as one scalar kernel's contraction" points
    ~f:(fun pt -> Float.(pt.Cal.ip_ms > 0.) && pt.Cal.ip_scalar_ops > 0 && pt.Cal.ip_mma_ops = 0);
  gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the kernel's operations grow with the reduction extent"
    (match points with [ a; b ] -> b.Cal.ip_scalar_ops > a.Cal.ip_scalar_ops | _ -> false);
  gated ~when_:(not seeds_here) ~on:backend_name
    "where the seeded default does not apply, the sweep refuses rather than time the default"
    ((not seeds_here)
    &&
    match sweep () with
    | _ -> false
    | exception Utils.User_error msg -> String.is_substring msg ~substring:"does not apply")
