(* gh-ocannl-1606: a test reads a node's debug name only to show it, never to find the node -- the
   debug name is a display name a renderer can change (gh-ocannl-1552). The reader, its closed list
   of accepted forms and what it deliberately does not read are in test/support/debug_name_scan.ml's
   header.

   Live: [debug_name_scan.exe <root> <generated>...] reads a dune sandbox holding the source tree
   and scans every .ml under test/ and arrayjit/test/. Control: [debug_name_scan.exe --fixture
   <root>] reads the same layout under a fixture tree, each file with a [.fixture] suffix so that no
   build or other scan reads it as a source, and ends as a negative control: its golden is the
   findings, and it must exit 1. Refusal control: [debug_name_scan.exe --refusal-control] routes
   each of this file's own refusals through a synthetic input and claims exactly that refusal fired
   (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Debug_name_scan
module Inventory = Test_utils.Source_inventory

let in_corpus path =
  (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/")
  && String.is_suffix path ~suffix:".ml"

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   refusal control can plant the defect. *)
let reading_or_refusal ~fail ~what read =
  match Test_utils.Scan_attempt.run read with
  | Scanned result -> Some result
  | Unparsed why ->
      fail
        (Printf.sprintf "%s does not parse as OCaml (%s), so its debug-name uses went unchecked"
           what why);
      None
  | Raised exn ->
      fail
        (Printf.sprintf
           "the debug-name reader raised %s on %s, which parses -- a defect in the scanner rather \
            than in the source"
           exn what);
      None

(* [sources] maps each relative path to its content. Returns each file with its report. *)
let run ~fail sources =
  let corpus = List.filter sources ~f:(fun (path, _) -> in_corpus path) in
  if List.is_empty corpus then
    fail "no test source is among the sources read -- the rule's source tree matches nothing";
  let reports =
    List.filter_map corpus ~f:(fun (file, source) ->
        Option.map
          (reading_or_refusal ~fail ~what:file (fun () -> Scan.scan ~file source))
          ~f:(fun report -> (file, report)))
  in
  List.iter reports ~f:(fun (_, (report : Scan.report)) -> List.iter report.findings ~f:fail);
  reports

let read_fixture root =
  let rec walk relative =
    let full = Stdlib.Filename.concat root relative in
    if Stdlib.Sys.is_directory full then
      Stdlib.Sys.readdir full |> Array.to_list |> List.sort ~compare:String.compare
      |> List.concat_map ~f:(fun entry ->
          walk (if String.is_empty relative then entry else relative ^ "/" ^ entry))
    else
      match String.chop_suffix relative ~suffix:".fixture" with
      | Some path -> [ (path, In_channel.read_all full) ]
      | None -> []
  in
  walk ""

(* Each refusal of this file, reached through a synthetic input: the observation is recorded only
   when exactly the one message the format names fired. *)
let refusal_control () =
  let source = "test/operations/debug_name_scan.ml" in
  let routed_case label ~format ~fragment run =
    let messages = ref [] in
    run (fun message -> messages := message :: !messages);
    let routed =
      match !messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  routed_case "an invalid OCaml source reaches the parse refusal"
    ~format:"%s does not parse as OCaml (%s), so its debug-name uses went unchecked"
    ~fragment:"test/bad.ml does not parse as OCaml" (fun fail ->
      ignore (run ~fail [ ("test/bad.ml", "let =") ] : _ list));
  routed_case
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the debug-name reader raised %s on %s, which parses -- a defect in the scanner rather than \
       in the source" ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses"
    (fun fail ->
      ignore
        (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted") : unit option));
  routed_case "a run without test sources reaches the empty-corpus refusal"
    ~format:"no test source is among the sources read -- the rule's source tree matches nothing"
    ~fragment:"no test source is among the sources read" (fun fail ->
      ignore (run ~fail [ ("tensor/tensor.ml", "let x = 1") ] : _ list));
  Test_utils.Refusal_control_manifest.print source

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--refusal-control" ] -> refusal_control ()
  | [ _; "--fixture"; root ] ->
      ignore (run ~fail:Verdict.fail (read_fixture root) : _ list);
      Verdict.exit_negative_control ()
  | _ :: root :: generated ->
      let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
      let sources =
        Inventory.select inventory ~f:in_corpus
        |> List.map ~f:(fun (file : Inventory.file) ->
            (file.path, In_channel.read_all file.on_disk))
      in
      let reports = run ~fail:Verdict.fail sources in
      let accepted =
        List.concat_map reports ~f:(fun (file, (report : Scan.report)) ->
            List.map report.accepted ~f:(fun (_, form) -> (file, form)))
      in
      (* The accepted positions the corpus uses: each must be read somewhere, so a reader gone blind
         (an empty corpus, a position it stopped recognising) changes this list. *)
      List.iter Scan.positions ~f:(fun form ->
          if List.exists accepted ~f:(fun (_, f) -> String.equal f form) then
            printf "accepted position in use: %s\n" form);
      (* Canaries, one per position, each a use a regressed reader would stop finding. *)
      let accepts file form =
        List.exists accepted ~f:(fun (f, g) -> String.equal f file && String.equal g form)
      in
      p "the scan reads test_tnode_namespaces' printed debug names"
        (accepts "test/operations/test_tnode_namespaces.ml" Scan.printed);
      p "the scan reads cc_kernel_outline's debug name in a claim label"
        (accepts "test/operations/cc_kernel_outline.ml" Scan.claim_label);
      p "the scan reads scope_over_materialized's refusal-message needle"
        (accepts "test/operations/scope_over_materialized.ml" Scan.needle);
      p "the scan reads rewind_to_last_closure's [@rendered_name] marker"
        (accepts "test/operations/rewind_to_last_closure.ml" Scan.marked);
      Test_utils.Refusal_control_manifest.print "debug_name_scan.ml"
  | _ -> Stdlib.exit 2
