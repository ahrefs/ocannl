(* gh-ocannl-1632: the gate every count claim over a physical search goes through
   ([test/support/tune_counts.ml]), and the report field it reads.

   A search drops a digest from its dedup set when that digest's window is refused, cut as hopeless
   or stopped by the budget, so an equivalent later seed is timed afresh rather than deduplicated
   (gh-ocannl-1484). [report.timings_retimed] counts those later attempts, so a claim about how many
   candidates were timed or deduplicated can read the one departure that moved it rather than infer
   it from the refusal counters.

   Pinned here: - The gate's shape, over synthetic reports: a window-reached count ([`Reached])
   departs on a re-timed digest and on a budget cut only, an admitted count ([`Admitted]) on every
   departure, and a report with none is clean for both. - The field, on an executed search: a 16x16
   matmul on cc deduplicates one seed against an earlier equivalent one on a quiet host; with every
   window forced to read as contended ([Autotune.with_contended_test_windows]) that earlier window
   leaves the dedup set, and the equivalent seed is re-timed instead. The report counts it, and its
   [search_done] line says so. - The negative control: in a child process, a count claim that would
   FAIL ([false]) goes through the gate on such a forced-contention report, for both kinds. It must
   be skipped -- announced on stderr and recorded as an environment skip -- and the child must exit
   0: a closed gate never evaluates the claim, and never passes it silently. The skip records stay
   the child's, so the sweep's skip coverage never sees a claim this test skips on every box. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Progress = Stderr_capture.Progress
open Verdict.Claims

(* --- The gate's shape. --- *)

let quiet = Autotune.no_search_report ~timing:Autotune.Isolated

let shapes =
  [
    ("a report with no departure", quiet, false, false);
    ( "a refused window",
      { quiet with Autotune.timings_contended = 1; timings_unreadable = 1 },
      false,
      true );
    ("a window cut as hopeless", { quiet with Autotune.timings_hopeless = 1 }, false, true);
    ("a re-timed digest", { quiet with Autotune.timings_retimed = 1 }, true, true);
    ("a budget cut", { quiet with Autotune.budget_cut = true }, true, true);
  ]

let () =
  p_all "each count kind's gate departs exactly on the departures that move that count" shapes
    ~f:(fun (what, r, reached, admitted) ->
      let departs count = not (List.is_empty (Tune_counts.departures count r)) in
      let ok = Bool.equal (departs `Reached) reached && Bool.equal (departs `Admitted) admitted in
      if not ok then
        Stdio.eprintf "%s: reached departs %b (want %b), admitted departs %b (want %b)\n%!" what
          (departs `Reached) reached (departs `Admitted) admitted;
      ok)

(* --- A forced-contention search whose equivalent seed is re-timed. --- *)

(* The seeding menu, the duplicate among it included, follows the cc vector width the host probes:
   pinned for every search of the process, before the first. *)
let () = Fissioned_chain.small_menu_envelope ()
let n = 16

let mv =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:(-1.5) ~stride:0.5)

(* One search of a fresh matmul on cc with every window forced to read as contended, and its
   progress lines. *)
let forced_search ~tag =
  let a = TDSL.ndarray mv ~label:[ tag ^ "_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray mv ~label:[ tag ^ "_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op c = a * b in
  let comp = Train.forward c in
  let report = ref None in
  let ctx, lines =
    Stderr_capture.lines (fun () ->
        Autotune.with_contended_test_windows (fun () ->
            fst
              (Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1
                 ~timing:Autotune.Isolated ~cache_dir:"" ~progress:true
                 ~report:(fun r -> report := Some r)
                 (Context.cpu ()) comp Ir.Indexing.Empty)))
  in
  Context.release ctx;
  let r = Option.value_exn ~here:[%here] !report in
  Stdio.eprintf
    "%s (not part of the golden): timed %d, refused %d, hopeless %d, re-timed %d, budget cut %b\n%!"
    tag r.Autotune.candidates_timed r.Autotune.timings_contended r.Autotune.timings_hopeless
    r.Autotune.timings_retimed r.Autotune.budget_cut;
  (r, Progress.events lines "search_done")

(* The claims the child gates; both would fail if evaluated. *)
let gated_labels =
  [
    (`Reached, "forced: a window-reached count claim is not evaluated");
    (`Admitted, "forced: an admitted count claim is not evaluated");
  ]

let child_mode = "forced-gate-child"

let () =
  match Array.to_list (Sys.get_argv ()) with
  | [ _; mode ] when String.equal mode child_mode ->
      let r, _ = forced_search ~tag:"child" in
      List.iter gated_labels ~f:(fun (count, label) ->
          Tune_counts.claim ~count ~tag:"child" [ r ] label false)
  | _ ->
      let r, dones = forced_search ~tag:"forced" in
      p "forced: the search refused windows and admitted none"
        (r.Autotune.timings_contended > 0 && r.Autotune.candidates_timed = 0);
      p "forced: an equivalent seed was re-timed instead of deduplicated"
        (r.Autotune.timings_retimed > 0);
      p "forced: its search_done line carries the re-timed count"
        (match dones with
        | [ line ] ->
            Option.equal Int.equal
              (Progress.int_field line "retimed")
              (Some r.Autotune.timings_retimed)
        | _ -> false);
      let ((status, out, err) as child) = Fresh_process.run [ child_mode ] in
      let exe = Stdlib.Filename.basename Stdlib.Sys.executable_name in
      let skipped_and_recorded (_, label) =
        String.is_substring out ~substring:(label ^ ": true")
        && String.is_substring err
             ~substring:(Printf.sprintf "SKIPPED on %s (vacuous): %s" Tune_counts.skipped_on label)
        && String.is_substring err
             ~substring:
               (String.concat ~sep:"\t" [ "OCANNL_TOOL_VERDICT_SKIP"; "environment"; exe; label ])
      in
      let exited_clean = Poly.equal status (Unix.WEXITED 0) in
      if not (exited_clean && List.for_all gated_labels ~f:skipped_and_recorded) then
        Fresh_process.report ~label:"forced-gate child" child;
      p "forced: the child gating claims that would fail exits clean" exited_clean;
      p_all "forced: each gated count claim is skipped as a recorded environment gap" gated_labels
        ~f:skipped_and_recorded
