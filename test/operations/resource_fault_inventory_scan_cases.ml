(* gh-ocannl-778: synthetic positive and negative controls for
   [Test_scan.Resource_fault_inventory_scan], exercising every refusal class beside the nearest
   valid inventory table. *)

open Base
open Verdict.Claims
module Scan = Test_scan.Resource_fault_inventory_scan

let points = [ "Point_a"; "Point_b" ]
let fi_exports = Set.of_list (module String) [ "with_callback"; "hit" ]
let autotune_exports = Set.of_list (module String) [ "on_candidate_measured"; "tune" ]

let existing_files =
  Set.of_list (module String) [ "test/operations/real_test.ml"; "arrayjit/test/other_test.ml" ]

let run doc =
  Scan.check ~points ~fi_exports ~autotune_exports ~file_exists:(Set.mem existing_files) doc

let valid_doc =
  {|# Inventory

| Seam | Failure mode | Status and acceptance evidence | Remaining leg |
| --- | --- | --- | --- |
| Seam A | Fails A | Covered: `test/operations/real_test.ml` hits `Point_a` and `Autotune.on_candidate_measured`. | None. |
| Seam B | Fails B | Covered: `arrayjit/test/other_test.ml` hits `Resource_fault_injection.Point_b` via `Resource_fault_injection.with_callback`. | None. |
| Seam C | Uncovered | Structural guard only. | Hardware leg pending. |
|}

let () =
  let ok = run valid_doc in
  p "valid inventory reports exact row count" (ok.data_rows = 3);
  p_empty "valid inventory passes with zero findings" ~over:points ok.findings;

  (* 1. Missing point constructor *)
  let missing_point =
    {|
| Seam | Failure mode | Status and acceptance evidence | Remaining leg |
| --- | --- | --- | --- |
| Seam A | Fails A | Covered: `test/operations/real_test.ml` hits `Point_a`. | None. |
|}
  in
  p "missing point constructor is refused"
    (List.exists (run missing_point).findings ~f:(function
      | Scan.Missing_point { point = "Point_b" } -> true
      | _ -> false));

  (* 2. Unknown qualified Resource_fault_injection member *)
  let unknown_fi =
    valid_doc
    ^ "| Seam D | Bad FI | Covered: `test/operations/real_test.ml` hits \
       `Resource_fault_injection.Retired_point`. | None. |\n"
  in
  p "retired or unknown Resource_fault_injection member is refused"
    (List.exists (run unknown_fi).findings ~f:(function
      | Scan.Unknown_fi_member { token = "Resource_fault_injection.Retired_point"; _ } -> true
      | _ -> false));

  (* 3. Renamed or missing test file *)
  let renamed_test =
    valid_doc
    ^ "| Seam E | Renamed | Covered: `test/operations/renamed_test.ml` hits `Point_a`. | None. |\n"
  in
  p "missing cited test file is refused"
    (List.exists (run renamed_test).findings ~f:(function
      | Scan.Missing_cited_file { path = "test/operations/renamed_test.ml"; _ } -> true
      | _ -> false));

  (* 4. Covered row without a cited test path *)
  let covered_no_test =
    valid_doc
    ^ "| Seam F | No path | Covered on `cc`: hits `Point_a` without naming a test file. | None. |\n"
  in
  p "Covered row without a cited test file is refused"
    (List.exists (run covered_no_test).findings ~f:(function
      | Scan.Covered_without_test _ -> true
      | _ -> false));

  (* 5. Unknown Autotune hook *)
  let bad_hook =
    valid_doc
    ^ "| Seam G | Bad hook | Covered: `test/operations/real_test.ml` uses \
       `Autotune.nonexistent_hook`. | None. |\n"
  in
  p "unknown Autotune hook is refused"
    (List.exists (run bad_hook).findings ~f:(function
      | Scan.Unknown_autotune_hook { token = "Autotune.nonexistent_hook"; _ } -> true
      | _ -> false));

  (* 6. Malformed row *)
  let malformed = valid_doc ^ "| Seam H | Too few cells |\n" in
  p "malformed pipe-table row is refused"
    (List.exists (run malformed).findings ~f:(function
      | Scan.Malformed_row { cells = 2; _ } -> true
      | _ -> false));

  (* 7. Unclosed backtick *)
  let unclosed =
    valid_doc
    ^ "| Seam I | Unclosed | Covered: `test/operations/real_test.ml` `unclosed | None. |\n"
  in
  p "unclosed backtick in table row is refused"
    (List.exists (run unclosed).findings ~f:(function
      | Scan.Unclosed_backtick _ -> true
      | _ -> false))
