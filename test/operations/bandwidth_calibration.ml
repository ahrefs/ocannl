(* End-to-end streaming memory-bandwidth calibration (gh-ocannl-578): [Ocannl.Calibrate.stream] over
   tiny tensors, on the configured backend. The pass's whole point is producing calibration rows
   with EXACT byte counts — the fitter's per-leg exactness rule bars approximate rows from the
   memory leg, which is why matmul-family tuning data alone leaves [model_peak_memory_bandwidth]
   unfittable. Timings are machine-dependent, so only structural facts are printed: rows were
   appended through the ordinary tuning emission path, bytes-exact rows are among them, and the fit
   over just these rows yields a memory-leg constant.

   The calibration file is pinned by the companion dune rule
   (--ocannl_autotune_calibration_file=bandwidth_calibration.tsv) and truncated here at start, so
   reruns in the same _build directory stay bounded and self-contained.

   WHICH kernels contribute rows is a property of the host, not of the pass (gh-ocannl-892). A
   timing window whose samples are mostly host stalls is refused ([Autotune.admitted_timing_ms]),
   and a refused candidate emits no row; on a busy machine a whole kernel's candidates can be
   refused, and then that kernel has no rows at all. So the golden pins the RELATIONSHIP -- a kernel
   has exactly as many rows as admitted timings, and a kernel with no rows shows refused timings as
   the evidence -- rather than an ordered list of the four names, which on the GPU backends made the
   golden a function of the load the sweep happened to be under. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cal = Ir.Cost_model.Calibration

let contributed rows name = List.exists rows ~f:(fun row -> String.equal row.Cal.routine name)
let row_count rows name = List.count rows ~f:(fun row -> String.equal row.Cal.routine name)

let row_count_matches_timing rows (name, report) =
  row_count rows name = report.Autotune.candidates_timed

(* Exercise the exact gh-ocannl-886 defect class against the same predicate as the live claim: start
   from a kernel that really timed and emitted, hide every one of its rows, and require the
   resulting timed-with-no-row observation to be rejected. Selecting from the live run keeps the
   control honest about the report schema without pinning which host-dependent kernel
   contributes. *)
let omission_control_rejected rows reports =
  List.find reports ~f:(fun (name, report) ->
      report.Autotune.candidates_timed > 0 && contributed rows name)
  |> Option.exists ~f:(fun ((name, _) as report) ->
      let rows_without_kernel =
        List.filter rows ~f:(fun row -> not (String.equal row.Cal.routine name))
      in
      not (row_count_matches_timing rows_without_kernel report))

(* A deterministic two-timing fixture keeps partial loss distinct from whole-kernel omission even
   when host contention leaves the live run with only one admitted timing per routine. Seed it with
   a real parsed row and report, then vary only multiplicity through the live predicate. *)
let count_controls rows reports =
  let fixture =
    List.find_map reports ~f:(fun (name, report) ->
        List.find rows ~f:(fun row -> String.equal row.Cal.routine name)
        |> Option.map ~f:(fun row -> (row, (name, { report with Autotune.candidates_timed = 2 }))))
  in
  Verdict.p "count control: two rows for two timings are accepted"
    (Option.exists fixture ~f:(fun (row, report) -> row_count_matches_timing [ row; row ] report));
  Verdict.p "count control: losing one of two rows is rejected"
    (Option.exists fixture ~f:(fun (row, report) -> not (row_count_matches_timing [ row ] report)));
  Verdict.p "count control: adding a spurious third row is rejected"
    (Option.exists fixture ~f:(fun (row, report) ->
         not (row_count_matches_timing [ row; row; row ] report)))

let () =
  let file =
    String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:"")
  in
  assert (not (String.is_empty file));
  if Stdlib.Sys.file_exists file then Stdlib.Sys.remove file;
  let ctx = Context.auto () in
  let reports = Calibrate.stream ~elems:65536 ~repeats:1 ctx in
  Stdio.printf "kernels tuned: %s\n" (String.concat ~sep:" " (List.map reports ~f:fst));
  let rows =
    if Stdlib.Sys.file_exists file then
      List.filter_map (Stdio.In_channel.read_lines file) ~f:Cal.of_line
    else []
  in
  Verdict.p "rows appended" (not (List.is_empty rows));
  (* Every row names the computation it timed (gh-ocannl-635) — the writer-side half of the schema,
     which only an end-to-end tuning run exercises: the name comes from the comp's block comment
     through [Autotune.tune]'s compiles. Without it the fitted memory-leg floor cannot say which
     stream kernel demonstrated it, and per-kernel rates have to be reconstructed outside the
     rows. *)
  let kernels = List.map reports ~f:fst in
  Verdict.p_all "every row names its routine" rows ~f:(fun r ->
      List.mem kernels r.Cal.routine ~equal:String.equal);
  List.iter reports ~f:(fun (name, rep) ->
      Stdio.eprintf
        "%s (not part of the golden): %d candidate(s) timed, %d timing(s) refused, %d candidate(s) \
         failed, %d row(s)\n"
        name rep.Autotune.candidates_timed rep.Autotune.timings_contended
        rep.Autotune.candidates_failed (row_count rows name));
  (* [Calibrate.stream] forces search with no cache replay. The baseline initializes [n_timed]
     exactly when its admitted timing is emitted; each admitted candidate increments it once and
     emits once. The TSV aggregates all segments into ONE row, not one row per segment. Refused
     timings do neither. Exact multiplicity catches partial omission and duplicate emission too. *)
  Verdict.p_all "each kernel row count equals its admitted timing count" reports
    ~f:(row_count_matches_timing rows);
  count_controls rows reports;
  Verdict.p "omission control: a timed kernel with no row is rejected"
    (omission_control_rejected rows reports);
  (* The count equality alone would also accept a kernel whose every candidate failed compile or
     dispatch ([candidates_failed]): nothing timed, nothing contributed, both counts zero. That is
     the loss of coverage the ordered list used to catch, and it is not what load does -- a refused
     timing window increments [timings_contended]. So a kernel may go row-less only on that
     evidence. Cache replay would zero both counters, but [Calibrate.stream] passes [~cache_dir:""],
     so every search here times live. *)
  Verdict.p_all "a kernel with no rows was refused its timings, not silently lost" reports
    ~f:(fun (name, rep) -> contributed rows name || rep.Autotune.timings_contended > 0);
  let exact_bytes =
    List.filter rows ~f:(fun r ->
        (not r.Cal.bytes_approx) && (not r.Cal.opaque) && r.Cal.bytes > 0
        && Float.(r.Cal.measured_ms > 0.))
  in
  Verdict.p "bytes-exact rows present" (not (List.is_empty exact_bytes));
  let fits = Cal.fit rows in
  Verdict.p "single-backend fit" (List.length fits = 1);
  Verdict.p "memory leg fitted from these rows"
    (List.exists fits ~f:(fun f -> Option.is_some f.Cal.fit_peak_memory_bandwidth))
