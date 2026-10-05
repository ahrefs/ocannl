(* gh-ocannl-1005: a deliberately refused Verdict run must inherit neither build-log stream. *)
open Base
open Stdio
module Scan = Test_utils.Refused_action_scan
module Inventory = Test_utils.Source_inventory

let report dune_files =
  let result = Scan.scan dune_files in
  List.iter result.problems ~f:Verdict.fail;
  List.iter result.refusals ~f:(printf "Verdict refusal run: %s\n");
  eprintf "Refusal stream scan: %d Dune files, %d Verdict refusal runs\n" (List.length dune_files)
    (List.length result.refusals);
  if not (Verdict.any_failed ()) then
    printf "OK: accepted Verdict failures inherit neither stream.\n"

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; path ] -> report [ ("fixture/dune", In_channel.read_all path) ]
  | _ :: root :: generated ->
      let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
      let dune_files =
        Inventory.select inventory ~f:(fun path ->
            String.equal (Stdlib.Filename.basename path) "dune")
        |> List.map ~f:(fun (file : Inventory.file) ->
            (file.path, In_channel.read_all file.on_disk))
      in
      if List.length dune_files < 3 then
        Verdict.fail (root ^ ": Dune source inventory below floor (3 files)");
      report dune_files;
      Test_utils.Refusal_control_manifest.print "refused_action_streams.ml"
  | _ -> Stdlib.exit 2
