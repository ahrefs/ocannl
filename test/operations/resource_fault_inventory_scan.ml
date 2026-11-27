(** gh-ocannl-778: enforces [docs/resource-fault-injection-inventory.md] against the derived
    [Ir.Resource_fault_injection.point] constructors, repository test files, and cited module hooks.

    The reader's closed boundary lives at the head of {!Test_scan.Resource_fault_inventory_scan}. *)

open Base
open Stdio
module Inventory = Test_scan.Source_inventory
module Scan = Test_scan.Resource_fault_inventory_scan

let inventory_doc = "docs/resource-fault-injection-inventory.md"
let fi_mli = "arrayjit/lib/resource_fault_injection.mli"
let autotune_mli = "arrayjit/lib/autotune.mli"
let min_rows = 10
let min_points = 15

let () =
  let root, generated =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: root :: generated -> (root, generated)
    | _ ->
        eprintf "Usage: resource_fault_inventory_scan root generated...\n";
        Stdlib.exit 2
  in
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let file_set =
    List.map (Inventory.files inventory) ~f:(fun f -> f.Inventory.path)
    |> Set.of_list (module String)
  in
  let read path =
    match List.find (Inventory.files inventory) ~f:(fun f -> String.equal f.path path) with
    | Some f -> In_channel.read_all f.on_disk
    | None -> failwith ("resource_fault_inventory_scan: missing required file " ^ path)
  in
  let points = Scan.derived_points () in
  let fi_exports = Scan.mli_exports (read fi_mli) in
  let autotune_exports = Scan.mli_exports (read autotune_mli) in
  let doc = read inventory_doc in
  let result =
    Scan.check ~points ~fi_exports ~autotune_exports ~file_exists:(Set.mem file_set) doc
  in
  List.iter result.findings ~f:(fun f -> eprintf "%s: %s\n" inventory_doc (Scan.describe f));
  eprintf "Scanned %s: %d table rows across %d derived fault-injection points.\n" inventory_doc
    result.data_rows (List.length points);
  Verdict.claim "derived fault-injection point census meets its floor"
    (List.length points >= min_points);
  Verdict.claim "inventory table row count meets its floor" (result.data_rows >= min_rows);
  Verdict.p_empty "every fault-injection point, cited test file, and cited hook resolves"
    ~over:points result.findings;
  Test_scan.Refusal_control_manifest.print "resource_fault_inventory_scan.ml"
