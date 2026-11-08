open Base
open Stdio
module Inventory = Test_scan.Source_inventory
module Scan = Test_scan.Markdown_link_scan

let () =
  let root, generated =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: root :: generated -> (root, generated)
    | _ ->
        eprintf "Usage: markdown_links root generated...\n";
        Stdlib.exit 2
  in
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let files = Inventory.select inventory ~f:Scan.in_corpus in
  Verdict.claim "the Markdown corpus includes docs and benchmark documents"
    (List.length files >= 100);
  List.iter [ "README.md"; "CHANGES.md" ] ~f:(fun path ->
      Verdict.claim ("the corpus includes " ^ path)
        (List.exists files ~f:(fun file -> String.equal file.path path)));
  List.iter [ "docs/"; "benchmarks/" ] ~f:(fun prefix ->
      Verdict.claim ("the corpus includes " ^ prefix)
        (List.exists files ~f:(fun file -> String.is_prefix file.path ~prefix)));
  (* Dune materializes source symlinks as regular files. Keep these known document aliases explicit
     and hold their contents equal to the canonical documents before resolving links. *)
  let aliases =
    [
      ("docs/research/imbue-infrastructure-lessons.md", "docs/imbue-infrastructure-lessons.md");
      ("docs/research/megakernel-deep-dive.md", "docs/megakernel-deep-dive.md");
    ]
  in
  List.iter aliases ~f:(fun (alias, canonical) ->
      let read path = In_channel.read_all (Stdlib.Filename.concat root path) in
      Verdict.claim "the documented symlink alias equals its canonical document"
        (String.equal (read alias) (read canonical)));
  let paths = List.map (Inventory.files inventory) ~f:(fun file -> file.path) in
  let findings =
    List.concat_map files ~f:(fun (file : Inventory.file) ->
        let from_file =
          Option.value (List.Assoc.find aliases file.path ~equal:String.equal) ~default:file.path
        in
        Scan.check ~exists:(Scan.path_exists ~files:paths) ~from_file
          (In_channel.read_all file.on_disk)
        |> List.map ~f:(fun (target, message) ->
            eprintf "%s: %s: %s\n" file.path target message;
            (file.path, target)))
  in
  eprintf "Scanned %d Markdown files.\n" (List.length files);
  Verdict.p_empty "every recognised relative Markdown link resolves" ~over:files findings;
  Test_scan.Refusal_control_manifest.print "markdown_links.ml"
