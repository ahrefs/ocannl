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

let scan ?(pin_harness = true) ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources =
    Inventory.select inventory ~f:Test_utils.Ll_test_scan.test_source
    |> List.map ~f:(fun (file : Inventory.file) -> (file.path, In_channel.read_all file.on_disk))
  in
  let read =
    List.concat_map sources ~f:(fun (path, content) ->
        List.map (Scan.tensorized_executed content) ~f:(fun (r : Scan.reading) ->
            (path, r.claim, r.judged)))
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
  (* The executor table pinned to the harness it names: a new Ll_test function reaching Context.run
     is an executor whether or not anyone remembered to list it. *)
  if pin_harness then (
    (* A harness missing from the inventory derives no executor, which the claim refuses. *)
    let derived =
      Option.value_map ~default:[] ~f:Scan.executors_of_harness
        (List.Assoc.find sources "test/support/ll_test.ml" ~equal:String.equal)
    in
    if not (List.equal String.equal derived Scan.ll_test_executors) then
      eprintf "Ll_test executors: listed %s, derived from test/support/ll_test.ml %s\n"
        (String.concat ~sep:" " Scan.ll_test_executors)
        (String.concat ~sep:" " derived);
    p "the Ll_test executor forms are exactly the harness functions that reach Context.run"
      (List.equal String.equal derived Scan.ll_test_executors));
  List.iter exemptions ~f:(fun (e : Scan.exemption) ->
      printf "%s -- `%s`: %s\n" e.path e.label e.reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~pin_harness:false ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan ~pin_harness:false
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
