(* gh-ocannl-1729: prose claiming the exactness of partial sums sits beside the claim deriving it.

   gh-ocannl-1693 derived the HIP bitwise legs' exactness premise with [Ll_test.exact_partials]
   instead of stating it in a comment, after prose rationales had the magnitudes wrong twice
   (gh-ocannl-1658). This holds every bitwise claim's premise to the same rule: a sentence of a test
   source claiming that exactness for its partial sums sits in a structure item that derives it.
   What the reader takes for such a sentence, and for "beside", is in
   test/support/exact_partials_scan.ml's header: it is a LINT over a fixed phrase list, and a missed
   phrasing is its gap, not a grant.

   The sites written before the rule are a ratchet: each row below holds one file's count of
   uncovered sentences EXACTLY, so converting one (deriving the premise with [exact_partials], or
   rewording a sentence whose claim was not about exactness) lowers the count in the same commit,
   and a new uncovered sentence anywhere fails. A row whose file has none left is stale and fails
   too. Converted when a file is next touched; the four sites gh-ocannl-1729 named are converted. *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Exact_partials_scan
module Dune = Test_scan.Dune_stanza_scan
module Sources = Test_scan.Config_key_scan

let exemptions : Scan.exemption list =
  let debt =
    "written before the rule: derive the premise with Ll_test.exact_partials when next touched"
  in
  [
    { Scan.path = "test/operations/accum_width.ml"; allowance = Uncovered 1; reason = debt };
    { Scan.path = "test/operations/back_to_back_runs.ml"; allowance = Uncovered 1; reason = debt };
    {
      Scan.path = "test/operations/deferred_hit_compiles.ml";
      allowance = Uncovered 1;
      reason = debt;
    };
    {
      Scan.path = "test/operations/discriminating_values.ml";
      allowance = Uncovered 1;
      reason =
        "the drift claim exhibits its serial partial sums directly; an every-association \
         derivation is the follow-up";
    };
    {
      Scan.path = "test/operations/fragment_scope_move_legality.ml";
      allowance = Uncovered 1;
      reason = debt;
    };
    {
      Scan.path = "test/operations/hardware_warp_shuffle.ml";
      allowance = Uncovered 1;
      reason = debt;
    };
    { Scan.path = "test/operations/matmul_interchange.ml"; allowance = Uncovered 2; reason = debt };
    { Scan.path = "test/operations/privatize_recheck.ml"; allowance = Uncovered 1; reason = debt };
    {
      Scan.path = "test/operations/scan_companion_coverage.ml";
      allowance = Uncovered 1;
      reason =
        "leg 4 derives z's premise in its own item; this note on z's operands sits apart from it";
    };
    { Scan.path = "test/operations/schedule_batch_grid.ml"; allowance = Uncovered 2; reason = debt };
    {
      Scan.path = "test/operations/schedule_contraction_nest.ml";
      allowance = Uncovered 2;
      reason = debt;
    };
    {
      Scan.path = "test/operations/schedule_ldmatrix_matmul.ml";
      allowance = Uncovered 1;
      reason = debt;
    };
    { Scan.path = "test/operations/schedule_pad.ml"; allowance = Uncovered 1; reason = debt };
    { Scan.path = "test/operations/split_reduce_pair.ml"; allowance = Uncovered 2; reason = debt };
    { Scan.path = "test/operations/tile_mma_narrow.ml"; allowance = Uncovered 2; reason = debt };
    { Scan.path = "test/operations/tune_arm_a_replay.ml"; allowance = Uncovered 1; reason = debt };
    {
      Scan.path = "test/operations/workgroup_memory_accounting.ml";
      allowance = Uncovered 1;
      reason = debt;
    };
    {
      Scan.path = "test/support/ll_test.ml";
      allowance = Uncovered 4;
      reason =
        "the harness's operand docs state the premises its helpers are built for, exact_partials' \
         own included; a test derives them at its use";
    };
    {
      Scan.path = "test/support/exact_partials_scan.ml";
      allowance = Every;
      reason = "the reader's own documentation quotes the phrases it reads";
    };
    {
      Scan.path = "test/operations/exact_partials_scan_cases.ml";
      allowance = Every;
      reason = "the reader's cases plant the phrases on synthetic sources";
    };
  ]

(* The sources arrive as a `@<path>` response file, as verdict_ratchet's do; see
   [Test_scan.Scan_argv]. *)
let argv = Test_scan.Scan_argv.expand Stdlib.Sys.argv

let () =
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" argv.(0);
    Stdlib.exit 2);
  let base = Dune.base_dir argv.(1) in
  (* Reported repository-relative, opened as dune handed them over. *)
  let arguments =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let sources =
    Sources.sources_among (List.map arguments ~f:fst)
    |> List.filter ~f:Test_scan.Ll_test_scan.test_source
  in
  let rows =
    List.filter_map sources ~f:(fun source ->
        match Scan.attempt ~source (In_channel.read_all (Map.find_exn on_disk source)) with
        | Ok reading -> Some (source, reading)
        | Error refusal ->
            Verdict.fail refusal;
            None)
  in
  let count f = List.sum (module Int) rows ~f:(fun (_, reading) -> f reading) in
  eprintf
    "exact-partials sentences read: %d covered, %d uncovered, in %d sources (not part of the golden)\n"
    (count (fun r -> List.length r.covered))
    (count (fun r -> List.length r.uncovered))
    (List.length rows);
  List.iter rows ~f:(fun (source, (reading : Scan.reading)) ->
      List.iter reading.uncovered ~f:(fun (c : Scan.claim) ->
          eprintf "  uncovered %s:%d: %s\n" source c.line c.sentence));
  let findings, stale = Scan.violations ~exemptions rows in
  List.iter (findings @ stale) ~f:Verdict.fail;
  p "every test source lexes and parses" (List.length rows = List.length sources);
  p_exists "the reader read prose beside a derivation in some test source" rows
    ~f:(fun (_, reading) -> not (List.is_empty reading.covered));
  p_exists "the walk read the arrayjit package's test sources too" rows ~f:(fun (path, _) ->
      String.is_prefix path ~prefix:"arrayjit/test/");
  p_empty
    "every sentence claiming the exactness of partial sums sits beside its derivation, or in its \
     file's exemption row"
    ~over:rows findings;
  p_empty "every exemption row is still earned" ~over:exemptions stale;
  List.iter exemptions ~f:(fun (e : Scan.exemption) ->
      match e.allowance with
      | Uncovered n -> printf "%s -- %d uncovered: %s\n" e.path n e.reason
      | Every -> printf "%s -- every sentence: %s\n" e.path e.reason);
  Test_scan.Refusal_control_manifest.print "exact_partials_ratchet.ml"
