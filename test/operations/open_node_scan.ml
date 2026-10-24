(* gh-ocannl-1579: every tensor node a module of tensor/ creates reaches Tensor.note_open_node, the
   uid list a rewind discards by (gh-ocannl-1505). The reader, its closed lists and what it
   deliberately does not read are in test/support/open_node_scan.ml's header. The creators are
   derived from arrayjit/lib/tnode.ml, never listed here.

   Live: [open_node_scan.exe <root> <generated>...] reads a dune sandbox holding tensor/'s source
   tree and arrayjit/lib/tnode.ml. Control: [open_node_scan.exe --fixture <root>] reads the same
   layout under a fixture tree, each file with a [.fixture] suffix so that no other scan or build
   reads it as a source, and ends as a negative control: its golden is the findings, and it must
   exit 1. Refusal control: [open_node_scan.exe --refusal-control] routes each of this file's own
   refusals through a synthetic input and claims exactly that refusal fired (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Open_node_scan
module Inventory = Test_utils.Source_inventory

let tnode_path = "arrayjit/lib/tnode.ml"
let in_corpus path = String.is_prefix path ~prefix:"tensor/" && String.is_suffix path ~suffix:".ml"

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   refusal control can plant the defect. *)
let reading_or_refusal ~fail ~what read =
  match Test_utils.Scan_attempt.run read with
  | Scanned result -> Some result
  | Unparsed why ->
      fail
        (Printf.sprintf "%s does not parse as OCaml (%s), so its creations went unchecked" what why);
      None
  | Raised exn ->
      fail
        (Printf.sprintf
           "the open-node reader raised %s on %s, which parses -- a defect in the scanner rather \
            than in the source"
           exn what);
      None

(* [sources] maps each relative path to its content; the tnode.ml entry derives the creators. *)
let run ~fail sources =
  match List.Assoc.find sources tnode_path ~equal:String.equal with
  | None ->
      fail
        (Printf.sprintf "%s is not among the sources read, so no creator can be derived" tnode_path);
      None
  | Some tnode -> (
      match
        reading_or_refusal ~fail ~what:tnode_path (fun () ->
            Scan.derive_creators ~file:tnode_path tnode)
      with
      | None -> None
      | Some creators ->
          List.iter creators.derivation_findings ~f:fail;
          let corpus = List.filter sources ~f:(fun (path, _) -> in_corpus path) in
          if List.is_empty corpus then
            fail
              "no tensor/ source is among the sources read -- the rule's source tree matches \
               nothing";
          let reports =
            List.filter_map corpus ~f:(fun (file, source) ->
                reading_or_refusal ~fail ~what:file (fun () -> Scan.scan ~file ~creators source))
          in
          List.iter reports ~f:(fun (report : Scan.report) -> List.iter report.findings ~f:fail);
          Some (creators, List.concat_map reports ~f:(fun report -> report.inventory)))

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
  let source = "test/operations/open_node_scan.ml" in
  let routed_case label ~format ~fragment run =
    let messages = ref [] in
    run (fun message -> messages := message :: !messages);
    let routed =
      match !messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  let tnode = "let fresh_uid () = 0\nlet create () = fresh_uid ()\n" in
  routed_case "an invalid OCaml source reaches the parse refusal"
    ~format:"%s does not parse as OCaml (%s), so its creations went unchecked"
    ~fragment:"tensor/bad.ml does not parse as OCaml" (fun fail ->
      ignore (run ~fail [ (tnode_path, tnode); ("tensor/bad.ml", "let =") ] : _ option));
  routed_case
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the open-node reader raised %s on %s, which parses -- a defect in the scanner rather than \
       in the source" ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses"
    (fun fail ->
      ignore
        (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted") : unit option));
  routed_case "a run without tnode.ml reaches the missing-derivation refusal"
    ~format:"%s is not among the sources read, so no creator can be derived"
    ~fragment:"arrayjit/lib/tnode.ml is not among the sources read" (fun fail ->
      ignore (run ~fail [ ("tensor/tensor.ml", "let x = 1") ] : _ option));
  routed_case "a run without tensor/ sources reaches the empty-corpus refusal"
    ~format:"no tensor/ source is among the sources read -- the rule's source tree matches nothing"
    ~fragment:"no tensor/ source is among the sources read" (fun fail ->
      ignore (run ~fail [ (tnode_path, tnode) ] : _ option));
  Test_utils.Refusal_control_manifest.print source

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--refusal-control" ] -> refusal_control ()
  | [ _; "--fixture"; root ] ->
      ignore (run ~fail:Verdict.fail (read_fixture root) : _ option);
      Verdict.exit_negative_control ()
  | _ :: root :: generated ->
      let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
      let sources =
        Inventory.select inventory ~f:(fun path -> String.equal path tnode_path || in_corpus path)
        |> List.map ~f:(fun (file : Inventory.file) ->
            (file.path, In_channel.read_all file.on_disk))
      in
      (match run ~fail:Verdict.fail sources with
      | None -> ()
      | Some (creators, followed) ->
          List.iter creators.names ~f:(fun name -> printf "creator: Tnode.%s\n" name);
          List.iter followed ~f:print_endline;
          (* Canaries: a scan gone blind and an intact tree print the same nothing. The issue's
             creator must be derived, and a creation must be followed through a pattern binding a
             pair and on through a match arm into the variable that is noted. *)
          p_exists "the scan derives Tnode.create as a creator" creators.names ~f:(fun name ->
              String.equal name "create");
          p_exists "the scan follows a pair-returning creation into a noted variable of tensor.ml"
            followed ~f:(fun line ->
              String.is_prefix line
                ~prefix:"tensor/tensor.ml: in op, Tnode.create_from_padded -> tn, init: noted"));
      Test_utils.Refusal_control_manifest.print "open_node_scan.ml"
  | _ -> Stdlib.exit 2
