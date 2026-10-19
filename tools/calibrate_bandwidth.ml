(* Memory-bandwidth calibration pass (gh-ocannl-578): run the STREAM-style footprint-exact kernels
   of [Ocannl.Calibrate.stream] on the configured backend, so the envelope's memory leg
   ([model_peak_memory_bandwidth]) becomes fittable — matmul-family tuning data alone cannot
   constrain it (compute-bound rows with approximate byte counts).

   Usage: dune exec tools/calibrate_bandwidth.exe -- [--elems=N] [--repeats=N] [--ocannl_* flags]

   Config [autotune_calibration_file] must be set (config file, OCANNL_AUTOTUNE_CALIBRATION_FILE, or
   --ocannl_autotune_calibration_file=PATH): the timed candidates append their rows there through
   the ordinary tuning emission path. Afterwards the pass reports each kernel's demonstrated
   bandwidth off those rows (they name the computation they timed), verifies bytes-exact rows
   appeared at all — failing loudly otherwise, rather than leaving the memory leg silently
   unfittable — and refits the whole file, printing config-pasteable [model_peak_*] constants for
   this backend (concatenating data from several runs only tightens the fit; use
   tools/fit_envelope.exe for multi-backend files or --margin headroom).

   [--elems] should stay a power of two (default 2^26, 256 MiB per stream tensor): an extent every
   workgroup size divides evenly keeps parallelized candidates free of range guards, hence
   bytes-exact. Backend and device follow the ordinary config (--ocannl_backend=...). Linking
   [ocannl] makes this a config consumer like any OCANNL executable; its startup chatter goes to
   stderr, leaving stdout a clean redirectable data channel. *)

open Base
module Cal = Ir.Cost_model.Calibration

let usage () =
  Stdio.eprintf "usage: calibrate_bandwidth [--elems=N] [--repeats=N] [--ocannl_* flags]\n";
  Stdlib.exit 2

let () =
  let args = List.tl_exn (Array.to_list Stdlib.Sys.argv) in
  let elems = ref (1 lsl 26) and repeats = ref None in
  List.iter args ~f:(fun a ->
      let int_flag prefix = Option.map (String.chop_prefix a ~prefix) ~f:Int.of_string in
      match int_flag "--elems=" with
      | Some n when n > 0 -> elems := n
      | Some _ | (exception _) -> usage ()
      | None -> (
          match int_flag "--repeats=" with
          | Some n when n > 0 -> repeats := Some n
          | Some _ | (exception _) -> usage ()
          | None ->
              (* Any spelling the config machinery accepts for a known key (--ocannl_backend=...,
                 --ocannl-backend=..., --backend=..., ...) is its argument, not ours; anything else
                 is a probable typo. *)
              if not (Utils.cmdline_arg_is_config_key a) then usage ()));
  let file = Utils.registered_text ~arg_name:"autotune_calibration_file" in
  if String.is_empty file then (
    Stdio.eprintf
      "calibrate_bandwidth: config autotune_calibration_file is not set — pass \
       --ocannl_autotune_calibration_file=PATH (or set it in the config file / environment)\n";
    Stdlib.exit 2);
  let read_lines () =
    if Stdlib.Sys.file_exists file then Stdio.In_channel.read_lines file else []
  in
  let before = List.length (read_lines ()) in
  let ctx = Context.auto () in
  let reports = Ocannl.Calibrate.stream ~elems:!elems ?repeats:!repeats ctx in
  let rows = List.filter_map (List.drop (read_lines ()) before) ~f:Cal.of_line in
  if List.is_empty rows then (
    Stdio.eprintf
      "calibrate_bandwidth: the pass appended no calibration rows to %s — was anything timed?\n"
      file;
    Stdlib.exit 1);
  let exact_bytes =
    List.filter rows ~f:(fun r ->
        (not r.Cal.bytes_approx) && (not r.Cal.opaque) && r.Cal.bytes > 0
        && Float.(r.Cal.measured_ms > 0.))
  in
  let gbps r = Float.of_int r.Cal.bytes /. (r.Cal.measured_ms *. 1e-3) /. 1e9 in
  let by_rate r1 r2 = Float.compare (gbps r1) (gbps r2) in
  (* Per-kernel rates read off the rows themselves (gh-ocannl-635): each row names the computation
     it timed, so the fastest bytes-exact row of a kernel is the bandwidth that kernel demonstrated
     — on the byte count the fit below will use, rather than a nominal streams-times-elems
     reconstruction that could agree with the fitter only by coincidence. The tuning winner's time
     comes alongside: it may belong to a candidate whose bytes are approximate, hence to no row
     here. *)
  List.iter reports ~f:(fun (name, rep) ->
      let ms = rep.Autotune.best_ms in
      let winner =
        if Float.(ms > 0.) && Float.is_finite ms then Printf.sprintf "%8.4f ms" ms else "      n/a"
      in
      match
        List.max_elt
          (List.filter exact_bytes ~f:(fun r -> String.equal r.Cal.routine name))
          ~compare:by_rate
      with
      | Some r ->
          Stdio.printf "# %-13s winner %s, best bytes-exact row %7.2f GB/s (%s, digest %s)\n" name
            winner (gbps r) r.Cal.label r.Cal.digest
      | None -> Stdio.printf "# %-13s winner %s, no bytes-exact row\n" name winner);
  (match List.max_elt exact_bytes ~compare:by_rate with
  | None ->
      (* The whole point of this pass is exact-bytes rows; producing none is a failure, not a quiet
         degradation (likely cause: range guards from an extent the schedule could not split evenly
         — keep --elems a power of two). *)
      Stdio.eprintf
        "calibrate_bandwidth: FAILED — none of the %d new rows has an exact byte count, so the \
         memory leg remains unfittable; try a power-of-two --elems\n"
        (List.length rows);
      Stdlib.exit 1
  | Some r ->
      Stdio.printf "# demonstrated bandwidth floor: %.2f GB/s (%s, digest %s)\n" (gbps r)
        (Cal.row_name r) r.Cal.digest);
  let backend = (List.hd_exn rows).Cal.backend in
  let all_rows = List.filter_map (read_lines ()) ~f:Cal.of_line in
  match List.filter (Cal.fit all_rows) ~f:(fun f -> String.equal f.Cal.fit_backend backend) with
  | [ fit ] -> Stdio.print_string (Cal.report fit)
  | _ ->
      Stdio.eprintf "calibrate_bandwidth: no fit for backend %s over %s\n" backend file;
      Stdlib.exit 1
