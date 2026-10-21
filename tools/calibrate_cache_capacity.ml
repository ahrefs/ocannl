(* Cache-capacity calibration pass (gh-ocannl-1253): sweep the untiled weight-gradient contraction
   of [Ocannl.Calibrate.working_set] over reduction extents on the configured backend, and print the
   [model_cache_capacity] the ranking score's working-set proxy implies -- the boundary between the
   extents whose operand re-reads are served from cache and those that pay DRAM for them.

   Usage: dune exec tools/calibrate_cache_capacity.exe -- [--m=N] [--n=N] [--ks=K1,K2,...]
   [--repeats=N] [--ocannl_* flags]

   Stdout is a table, one row per extent -- the working set and per-block requested bytes the cost
   model reads off the scheduled kernel, the distinct bytes, the best time, the requested and
   distinct rates, and the ranking score's prediction under the capacity in force (rank) beside the
   distinct-cell roofline (bound) -- then a config-pasteable [model_cache_capacity=] line under [#]
   comment lines, with the [model_cache_bandwidth=] the below-capacity plateau implies -- or a [#]
   line saying why the sweep implies none. When no point of the sweep reaches DRAM only the cache
   bandwidth is printed: the capacity is then a lower bound, which no config value expresses.
   Calibrate on an otherwise idle device: the fit reads rates. Backend and device follow the
   ordinary config (--ocannl_backend=...); startup chatter goes to stderr. *)

open Base
module Cal = Ocannl.Calibrate

let usage () =
  Stdio.eprintf
    "usage: calibrate_cache_capacity [--m=N] [--n=N] [--ks=K1,K2,...] [--repeats=N] [--ocannl_* \
     flags]\n";
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
  let limits = Context.hardware_limits ctx in
  let capacity, resident = Autotune.ranking_capacity ~limits in
  let points = Cal.working_set ?m:!m ?n:!n ?ks:!ks ?repeats:!repeats ctx in
  let opt_int = Option.value_map ~default:"none" ~f:Int.to_string in
  Stdio.printf "# backend %s: capacity in force %s bytes, resident threads %s\n"
    (Context.backend_name ctx) (opt_int capacity) (opt_int resident);
  Stdio.printf "# %8s %14s %14s %12s %10s %12s %12s %10s %10s\n" "k" "working_set" "requested"
    "distinct" "ms" "req_GB/s" "dist_GB/s" "rank_ms" "bound_ms";
  Option.iter (List.hd points) ~f:(fun p ->
      Stdio.printf "# lowered loop order of the contraction (outermost first): [%s]\n"
        (String.concat ~sep:"," (List.map p.Cal.wp_loops ~f:Int.to_string)));
  List.iter points ~f:(fun p ->
      let gbps bytes = Float.of_int bytes /. (p.Cal.wp_ms *. 1e-3) /. 1e9 in
      let ms = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.4f") in
      Stdio.printf "  %8d %14d %14d %12d %10.4f %12.2f %12.2f %10s %10s\n" p.Cal.wp_k
        p.Cal.wp_working_set p.Cal.wp_requested p.Cal.wp_distinct p.Cal.wp_ms
        (gbps p.Cal.wp_requested) (gbps p.Cal.wp_distinct) (ms p.Cal.wp_rank_ms)
        (ms p.Cal.wp_bound_ms));
  (* The DRAM reference the fit splits against: the ranking score's own DRAM rate where one is in
     force (gh-ocannl-1442: config [model_dram_bandwidth] or the backend's advisory, which the
     score's DRAM tier divides by), else the envelope's memory leg as configured (pass the
     STREAM-calibrated [--ocannl_model_dram_bandwidth]; the envelope's advisory value is a class
     ceiling, far above any one device's DRAM). *)
  let dram_rate =
    match Autotune.ranking_dram_bandwidth ~limits with
    | Some r -> Some r
    | None -> Option.map (snd (Autotune.envelope_legs ~limits)) ~f:fst
  in
  Stdio.printf "# DRAM reference: %s bytes/s\n"
    (Option.value_map dram_rate ~default:"none (splitting on the sweep's own contrast)"
       ~f:(Printf.sprintf "%.6g"));
  match Cal.fit_capacity ?dram_rate points with
  | Cal.Transition { cached_ws; spilled_ws; capacity; monotone; cache_rate } ->
      Stdio.printf "# cached up to a working set of %d bytes, spilled from %d bytes%s\n" cached_ws
        spilled_ws
        (if monotone then "" else " -- NOT MONOTONE: the rates do not split by working set");
      Stdio.printf "model_cache_capacity=%d\nmodel_cache_bandwidth=%.6e\n" capacity cache_rate;
      if not monotone then Stdlib.exit 1
  | Cal.All_cached { largest_ws; cache_rate } ->
      (* A lower bound is not a capacity: no [model_cache_capacity] line, so nothing downstream
         reads "at least this" as "unlimited" (which [none] means) or as "exactly this". *)
      Stdio.printf
        "# every point is cache-served: the capacity is at least %d bytes, the largest swept \
         working set -- extend --ks to find it; no capacity is printed\n"
        largest_ws;
      Stdio.printf "model_cache_bandwidth=%.6e\n" cache_rate
  | Cal.All_spilled { smallest_ws } ->
      Stdio.printf
        "# every point is DRAM-bound: the capacity is below %d bytes -- extend --ks downward\n"
        smallest_ws;
      Stdlib.exit 1
  | Cal.No_transition { contrast } ->
      Stdio.printf
        "# no transition: requested rates differ by only %.2fx and no DRAM reference was given\n"
        contrast;
      Stdlib.exit 1
  | Cal.Too_few_points ->
      Stdio.printf "# too few timed points to fit\n";
      Stdlib.exit 1
