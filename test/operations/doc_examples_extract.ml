open Base
open Stdio
module Inventory = Test_utils.Source_inventory
module Examples = Test_utils.Doc_examples_scan

let () =
  let root = Stdlib.Sys.argv.(1) in
  let inventory =
    Inventory.of_dune_sandbox ~workspace_root:root
      ~generated:
        [
          Stdlib.Sys.argv.(0);
          "ocannl_config";
          "doc_examples.ml";
          "doc_examples.coverage";
          "doc_examples.inventory";
        ]
  in
  let blocks =
    Inventory.select inventory ~f:(fun path ->
        String.is_suffix path ~suffix:".md"
        && (String.is_prefix path ~prefix:"docs/"
           || String.equal path "AGENTS.md" || String.equal path "README.md"))
    |> List.concat_map ~f:(fun (file : Inventory.file) ->
        Examples.parse ~path:file.path (In_channel.read_all file.on_disk))
  in
  Examples.require_selected blocks;
  Out_channel.write_all "doc_examples.ml" ~data:(Examples.render blocks);
  Out_channel.write_all "doc_examples.coverage" ~data:(Examples.coverage blocks ^ "\n");
  Out_channel.write_all "doc_examples.inventory"
    ~data:
      (String.concat ~sep:"\n"
         (List.map blocks ~f:(fun block ->
              let status =
                match block.Examples.status with
                | Check id -> "checked " ^ id
                | Skip reason -> "excluded " ^ reason
                | Unchecked -> "unchecked: no annotation"
              in
              Printf.sprintf "%s:%d: %s" block.path block.line status))
      ^ "\n")
