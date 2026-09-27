(** The inventory of every file that names one of the virtualizer's rejection codes
    (gh-ocannl-1015), and the checklist a change to PIPELINE ORDER reads.

    Whether such a code can fire is decided by which passes run before the one that raises it, not
    by the raise site: gh-ocannl-483 put the algebraic rewrite tier ahead of [Low_level.optimize],
    one of its rewrites emits a hoisted local, and a refusal every description called defensive
    became reachable on the ordinary path. Its "cannot fire" claim then had six live copies across
    the library, the lowering doc, an agent note and a test header; the PR that moved the pass
    updated none of them, and the sweep that found them later missed one twice. This lists them.

    The codes are DERIVED, never listed here: {!Test_utils.Nonvirtual_code_scan} reads the library
    sources for the tag literals in the scope of the local exception, under the function that
    declares it -- its header states what a code and a mention are, and what is not read. The golden
    holds the minted codes by function, then every file naming one with the codes it names. A new
    code, a new file citing one, or a file that stops citing one each moves a line.

    Four refusals ride along, each on stderr and a nonzero exit. A citation of a code no raise site
    mints is stale (a retired code, or a tag of another family spelled as one). A number minted
    under two tags makes every numeric citation ambiguous; a tag minted in two functions makes the
    provenance a node records unable to say which phase refused it. And the boundary test's phase
    table -- the one place the phase of a code is written down -- must place each tag in the phase
    whose function mints it, which is what makes that table a restatement under test rather than
    beside the source. [phases] below is the bridge between the test's vocabulary and the functions.
*)

open Base
open Stdio
module Scan = Test_utils.Nonvirtual_code_scan
module Inventory = Test_utils.Source_inventory

let library_root = "arrayjit/lib/"
let table_source = "test/operations/virtual_rejection_boundary.ml"

(* This scan's own golden names every code by construction, so listing it would say nothing. *)
let own_golden = "test/operations/nonvirtual_code_inventory.expected"

(* The boundary test's phase constructors, and the function whose codes each one names. The cap
   phase is absent on purpose: the test derives it from [Low_level.is_cap_provenance], and caps are
   provenance constructors rather than codes. *)
let phases = [ ("Store", "check_and_store_virtual"); ("Consumption", "inline_computation") ]

(* Read, but not part of the checklist: text that describes the code as it stood on a date, and that
   a later change does not revise. A prefix no file lives under is stale and fails. *)
let records =
  [
    ("docs/proposals/", "design proposals, each describing the code as it stood when written");
    ("docs/in-progress/", "links into docs/proposals/, the proposals currently being worked on");
  ]

let scan ~records root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let read (file : Inventory.file) = In_channel.read_all file.on_disk in
  let codes =
    Inventory.select inventory ~f:(fun path ->
        String.is_prefix path ~prefix:library_root && String.is_suffix path ~suffix:".ml")
    |> List.concat_map ~f:(fun file ->
        let content = read file in
        if String.is_substring content ~substring:Scan.constructor then
          Scan.minted ~source:file.path content
        else [])
    |> Scan.merge
  in
  let in_record path = List.exists records ~f:(fun (prefix, _) -> String.is_prefix path ~prefix) in
  let files =
    List.filter_map (Inventory.files inventory) ~f:(fun file ->
        let content = read file in
        if
          in_record file.path || String.equal file.path own_golden || String.contains content '\000'
        then None
        else
          match Scan.mentions ~codes content with
          | [], [] -> None
          | named, unknown -> Some (file.path, named, unknown))
  in
  let table =
    Option.bind
      (List.find (Inventory.files inventory) ~f:(fun file -> String.equal file.path table_source))
      ~f:(fun file -> Scan.phase_table (read file))
  in
  let paths = List.map (Inventory.files inventory) ~f:(fun (file : Inventory.file) -> file.path) in
  List.iter
    (Scan.violations ~codes ~table_source ~table ~phases ~files @ Scan.stale_records ~records paths)
    ~f:Verdict.fail;
  printf
    "Codes, derived: each tag literal in the scope of a local `%s` exception, under the function \
     declaring it.\n"
    Scan.constructor;
  let by_origin (a : Scan.code) (b : Scan.code) =
    match String.compare a.source b.source with 0 -> String.compare a.minter b.minter | c -> c
  in
  List.stable_sort codes ~compare:by_origin
  |> List.group ~break:(fun a b -> by_origin a b <> 0)
  |> List.iter ~f:(fun group ->
      let first : Scan.code = List.hd_exn group in
      printf "%s, %s:\n" first.source first.minter;
      List.iter group ~f:(fun (c : Scan.code) -> printf "  %s\n" c.tag));
  printf "The phase table in %s is held to those functions:\n" table_source;
  List.iter phases ~f:(fun (phase, minter) -> printf "  %s -- %s\n" phase minter);
  printf
    "Files naming a code, as `%s N` or by its tag -- the checklist for a change to where a code \
     can fire:\n"
    Scan.constructor;
  List.iter files ~f:(fun (path, named, _) ->
      printf "%s --%s\n" path (String.concat (List.map named ~f:(fun n -> " " ^ Int.to_string n))));
  printf "Read, not part of the checklist:\n";
  List.iter records ~f:(fun (prefix, reason) -> printf "  %s -- %s\n" prefix reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~records:[] root []
  | _ :: root :: generated -> scan ~records root generated
  | _ -> Stdlib.exit 2
