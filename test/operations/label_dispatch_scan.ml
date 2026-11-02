(* gh-ocannl-1783: the compiler under arrayjit/lib/ reads a node's label or namespace, and a
   backend's name, only to show them, to copy a label onto a node it mints, or at a site whose
   marker says why the read decides nothing -- both are public spellings a user's tensor or a
   backend can forge (gh-ocannl-1721, gh-ocannl-1722, gh-ocannl-1723). The reading, its closed list
   of accepted forms and what it deliberately does not read are in
   test/support/label_dispatch_scan.ml's header; the walker is Debug_name_scan's.

   Live: [label_dispatch_scan.exe <root> <generated>...] reads a dune sandbox holding the source
   tree and scans every .ml under arrayjit/lib/. Control: [label_dispatch_scan.exe --fixture <root>]
   reads the same layout under a fixture tree, each file with a [.fixture] suffix so that no build
   or other scan reads it as a source, and ends as a negative control: its golden is the findings,
   and it must exit 1. Refusal control: [label_dispatch_scan.exe --refusal-control] routes each of
   this file's own refusals through a synthetic input and claims exactly that refusal fired
   (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Label_dispatch_scan
module Inventory = Test_utils.Source_inventory

let corpus_root = "arrayjit/lib/"

let in_corpus path =
  String.is_prefix path ~prefix:corpus_root && String.is_suffix path ~suffix:".ml"

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   refusal control can plant the defect. *)
let reading_or_refusal ~fail ~what read =
  match Test_utils.Scan_attempt.run read with
  | Scanned result -> Some result
  | Unparsed why ->
      fail
        (Printf.sprintf "%s does not parse as OCaml (%s), so its label reads went unchecked" what
           why);
      None
  | Raised exn ->
      fail
        (Printf.sprintf
           "the label-dispatch reader raised %s on %s, which parses -- a defect in the scanner \
            rather than in the source"
           exn what);
      None

(* [sources] maps each relative path to its content. Returns each file with its report. *)
let run ~fail sources =
  let corpus = List.filter sources ~f:(fun (path, _) -> in_corpus path) in
  if List.is_empty corpus then
    fail
      "no source under arrayjit/lib/ is among the sources read -- the rule's source tree matches \
       nothing";
  let reports =
    List.filter_map corpus ~f:(fun (file, source) ->
        Option.map
          (reading_or_refusal ~fail ~what:file (fun () -> Scan.scan ~file source))
          ~f:(fun report -> (file, report)))
  in
  List.iter reports ~f:(fun (_, (report : Test_utils.Debug_name_scan.report)) ->
      List.iter report.findings ~f:fail);
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
  let source = "test/operations/label_dispatch_scan.ml" in
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
    ~format:"%s does not parse as OCaml (%s), so its label reads went unchecked"
    ~fragment:"arrayjit/lib/bad.ml does not parse as OCaml" (fun fail ->
      ignore (run ~fail [ ("arrayjit/lib/bad.ml", "let =") ] : _ list));
  routed_case
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the label-dispatch reader raised %s on %s, which parses -- a defect in the scanner rather \
       than in the source" ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses"
    (fun fail ->
      ignore
        (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted") : unit option));
  routed_case "a run without corpus sources reaches the empty-corpus refusal"
    ~format:
      "no source under arrayjit/lib/ is among the sources read -- the rule's source tree matches \
       nothing" ~fragment:"no source under arrayjit/lib/ is among the sources read" (fun fail ->
      ignore (run ~fail [ ("lib/train.ml", "let x = 1") ] : _ list));
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
        List.concat_map reports ~f:(fun (file, (report : Test_utils.Debug_name_scan.report)) ->
            List.map report.accepted ~f:(fun (_, form) -> (file, form)))
      in
      (* The accepted positions the corpus uses: each must be read somewhere, so a reader gone blind
         (an empty corpus, a position it stopped recognising) changes this list. *)
      List.iter Scan.positions ~f:(fun form ->
          if List.exists accepted ~f:(fun (_, f) -> String.equal f form) then
            printf "accepted position in use: %s\n" form);
      (* Canaries, one per position this rule adds, each a use a regressed reader would stop
         finding. *)
      let accepts file form =
        List.exists accepted ~f:(fun (f, g) -> String.equal f file && String.equal g form)
      in
      let module D = Test_utils.Debug_name_scan in
      p "the scan reads online_softmax's tile labels copied onto new nodes"
        (accepts "arrayjit/lib/online_softmax.ml" D.copied);
      p "the scan reads backends' [%log] payloads" (accepts "arrayjit/lib/backends.ml" D.logged);
      p "the scan reads context's ~backend_name forwarding"
        (accepts "arrayjit/lib/context.ml" D.forwarded);
      p "the scan reads tnode's [@label_read] markers"
        (accepts "arrayjit/lib/tnode.ml" D.label_read);
      p "the scan reads memory_budget's [@message_only] log lines"
        (accepts "arrayjit/lib/memory_budget.ml" D.message_only);
      p "the scan reads low_level's composed refusal messages"
        (accepts "arrayjit/lib/low_level.ml" D.composed);
      Test_utils.Refusal_control_manifest.print "label_dispatch_scan.ml"
  | _ -> Stdlib.exit 2
