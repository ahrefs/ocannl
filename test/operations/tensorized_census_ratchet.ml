(* gh-ocannl-1309: an executed parity claim about a tensorized routine needs a census judgment
   beside it, since a Tile_mma that declined to the scalar fallback passes parity too. The reader,
   its closed lists of recognised forms and its boundary are in
   test/support/tensorized_census_scan.ml's header. *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Tensorized_census_scan
module Inventory = Test_utils.Source_inventory

(* A row absorbs every unjudged tensorized parity claim of its file with this label (the label
   expression as the refusal prints it), and says why that claim needs no census. A row nothing
   answers to is stale and fails. The tree needs none today. *)
let exemptions : Scan.exemption list = []

let scan ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources =
    Inventory.select inventory ~f:Test_utils.Ll_test_scan.test_source
    |> List.map ~f:(fun (file : Inventory.file) -> (file.path, In_channel.read_all file.on_disk))
  in
  let read =
    List.concat_map sources ~f:(fun (path, content) ->
        List.map (Scan.tensorized_executed content) ~f:(fun (claim, judged) ->
            (path, claim, judged)))
  in
  (* The population, for a reader of the log; a count on stdout would move with every addition. *)
  eprintf "executed tensorized parity claims read: %d\n" (List.length read);
  List.iter read ~f:(fun (path, (claim : Scan.claim), judged) ->
      eprintf "  %s:%d %s %s\n" path claim.line
        (if judged then "judged" else "UNJUDGED")
        claim.label);
  List.iter (Scan.violations ~exemptions sources) ~f:Verdict.fail;
  p_exists "the scan read an executed tensorized parity claim whose item judges the census" read
    ~f:(fun (_, _, judged) -> judged);
  List.iter exemptions ~f:(fun (e : Scan.exemption) ->
      printf "%s -- `%s`: %s\n" e.path e.label e.reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan
        ~exemptions:
          [
            {
              Scan.path = "test/new.ml";
              label = "\"tensorized matmul matches the serial twin\"";
              reason = "control exemption";
            };
          ]
        root []
  | _ :: root :: generated ->
      scan ~exemptions root generated;
      Test_utils.Refusal_control_manifest.print "tensorized_census_ratchet.ml"
  | _ -> Stdlib.exit 2
