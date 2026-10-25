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
    ip_contexts = [];
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

(* gh-ocannl-1479: the launch-fill fit. Synthetic sweeps generated from the capped fill at P = 70, t
   = 128 and a 4e12 full-fill rate, over two block widths (64 and 256 threads) from 1 block to many:
   the fit recovers t, the rate and the intercept, the same points' raw-operation line reads a
   fill-depressed rate (the pre-fill fit's reading), and the uncapped form fits worse. A sweep that
   never leaves full fill identifies no saturation, and tile-MMA work or too few points refuse. *)
let () =
  let np = 70 and t = 128 and rate = 4e12 and latency = 0.01 in
  let context ~blocks ~threads ~ops ~tile =
    { Ir.Cost_model.lc_blocks = blocks; lc_threads = threads; lc_tile_mma = tile; lc_ops = ops }
  in
  let lpoint ?(tile = false) ~blocks ~threads ops =
    let contexts = [ context ~blocks ~threads ~ops ~tile ] in
    let eff = Cal.effective_ops ~multiprocessors:np ~fill_threads:t contexts in
    {
      Cal.lp_geometry = Printf.sprintf "t%d" threads;
      lp_m = blocks;
      lp_n = 1;
      lp_k = 1;
      lp_contexts = contexts;
      lp_ms = eff /. rate *. 1e3;
    }
  in
  let sweep =
    List.map [ 1; 4; 16; 64; 256; 1024 ] ~f:(fun b ->
        lpoint ~blocks:b ~threads:64 ((1 lsl 31) + (b * 4096)))
    @ List.map [ 1; 4; 16; 64; 256 ] ~f:(fun b ->
        lpoint ~blocks:b ~threads:256 ((1 lsl 30) + (b * 8192)))
  in
  let rel a b = Float.(abs (a -. b) <= 1e-6 *. abs b) in
  p "the fill fit recovers the saturating threads and the full-fill rate"
    (match Cal.fit_fill ~multiprocessors:np sweep with
    | Cal.Fill_fit f -> f.saturation = t && rel f.rate rate && Float.(f.max_residual < 1e-6)
    | Cal.Fill_unfittable _ -> false);
  (* The pre-fill fit over a reduction-extent sweep of one partially filled launch (16 blocks of 64
     threads: fill 1024 / 8960) reads the full-fill rate times that fill -- the depressed rate the
     launch fit normalizes. *)
  let partial_fill = 1024. /. 8960. in
  let ksweep =
    List.map [ 1; 2; 4; 8 ] ~f:(fun g ->
        let ops = g * (1 lsl 28) in
        {
          (point ~ops ~ms:(latency +. (Float.of_int ops /. partial_fill /. rate *. 1e3))) with
          Cal.ip_contexts = [ context ~blocks:16 ~threads:64 ~ops ~tile:false ];
        })
  in
  p "the pre-fill fit over a partially filled launch reads the full-fill rate times its fill"
    (match Cal.fit_issue_rate ksweep with
    | Cal.Issue_fit { rate = raw; _ } -> rel raw (rate *. partial_fill) && Float.(raw < 0.5 *. rate)
    | Cal.Issue_unfittable _ -> false);
  p "under the fill model the same sweep reads the full-fill rate"
    (match
       Cal.fit_issue_rate
         ~fill:{ Ir.Cost_model.fm_multiprocessors = np; fm_fill_threads = t }
         ksweep
     with
    | Cal.Issue_fit { rate = normalized; _ } -> rel normalized rate
    | Cal.Issue_unfittable _ -> false);
  p "the uncapped form fits the capped sweep worse"
    (match Cal.fit_fill ~multiprocessors:np sweep with
    | Cal.Fill_fit { uncapped = Some (_, r); max_residual; _ } -> Float.(r > max_residual)
    | _ -> false);
  p "a sweep that never leaves full fill identifies no saturation"
    (match
       Cal.fit_fill ~multiprocessors:np
         (List.map [ 1024; 2048; 4096; 8192 ] ~f:(fun b ->
              lpoint ~blocks:b ~threads:256 ((1 lsl 30) + (b * 8192))))
     with
    | Cal.Fill_unfittable _ -> true
    | Cal.Fill_fit _ -> false);
  p "a point with tile-MMA work refuses: the fill is a scalar fit"
    (match
       Cal.fit_fill ~multiprocessors:np (lpoint ~tile:true ~blocks:4 ~threads:64 (1 lsl 20) :: sweep)
     with
    | Cal.Fill_unfittable _ -> true
    | Cal.Fill_fit _ -> false);
  p "a single geometry whose every point is partial identifies no saturation"
    (match
       Cal.fit_fill ~multiprocessors:np
         (List.map [ 1; 2; 4; 8 ] ~f:(fun b -> lpoint ~blocks:b ~threads:64 (1 lsl 30)))
     with
    | Cal.Fill_unfittable _ -> true
    | Cal.Fill_fit _ -> false);
  p "the rate is the caller's lead geometry's, whatever the series' names sort to"
    (let scaled =
       List.map sweep ~f:(fun lp ->
           if String.equal lp.Cal.lp_geometry "t256" then { lp with lp_ms = 2. *. lp.Cal.lp_ms }
           else lp)
     in
     let t256_first =
       List.filter scaled ~f:(fun lp -> String.equal lp.Cal.lp_geometry "t256")
       @ List.filter scaled ~f:(fun lp -> String.equal lp.Cal.lp_geometry "t64")
     in
     match
       (Cal.fit_fill ~multiprocessors:np scaled, Cal.fit_fill ~multiprocessors:np t256_first)
     with
     | Cal.Fill_fit a, Cal.Fill_fit b -> rel a.rate rate && rel b.rate (rate /. 2.)
     | _ -> false);
  p "fewer than three points refuse"
    (match Cal.fit_fill ~multiprocessors:np (List.take sweep 2) with
    | Cal.Fill_unfittable _ -> true
    | Cal.Fill_fit _ -> false)

let gpu_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"
  || String.is_substring backend_name ~substring:"metal"

let () =
  let sizes = [ 64; 128 ] in
  let sweep () =
    Cal.launch_sweep
      ~geometries:[ (32, 32, 8, 4, 4) ]
      ~sizes ~target_ops:(1 lsl 20) ~repeats:1 (Context.auto ())
  in
  let points = if gpu_here then sweep () else [] in
  gated_all ~min:(List.length sizes) ~when_:gpu_here ~on:backend_name
    "on a GPU backend every launch point is timed, one scalar context of 64-thread blocks" points
    ~f:(fun lp ->
      Float.(lp.Cal.lp_ms > 0.)
      &&
      match List.filter lp.Cal.lp_contexts ~f:(fun c -> c.Ir.Cost_model.lc_ops > 0) with
      | [ c ] -> c.Ir.Cost_model.lc_threads = 64 && not c.lc_tile_mma
      | _ -> false);
  gated ~when_:gpu_here ~on:backend_name "on a GPU backend the launch's blocks grow with the output"
    (match points with
    | [ a; b ] -> (
        match (a.Cal.lp_contexts, b.Cal.lp_contexts) with
        | ca :: _, cb :: _ -> cb.Ir.Cost_model.lc_blocks = 4 * ca.Ir.Cost_model.lc_blocks
        | _ -> false)
    | _ -> false);
  gated ~when_:(not gpu_here) ~on:backend_name
    "without the GPU sketch family the launch sweep refuses"
    ((not gpu_here)
    &&
    match sweep () with
    | _ -> false
    | exception Utils.User_error msg -> String.is_substring msg ~substring:"no GPU sketch family")
