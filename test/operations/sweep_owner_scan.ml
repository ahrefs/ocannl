(* gh-ocannl-926: every `Outside_sweep skip names the matrix that owns its execution, and that
   matrix still exists -- a CI step that builds the test's alias, or the plain run of the executable
   a rule forces into the skip. The reader, its closed list of recognised forms and what it
   deliberately does not read are in test/support/sweep_owner_scan.ml's header.

   [sweep_owner_scan.exe <root> <generated>...] reads a dune sandbox holding the source tree. Its
   golden is one line per owner and how it resolved; each refusal is a failed check. The reader's
   refusals are pinned on synthetic trees by sweep_owner_scan_cases, and this file's own by
   [sweep_owner_scan.exe --refusal-control], which routes each through a synthetic input and claims
   exactly that refusal fired (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Sweep_owner_scan
module Inventory = Test_scan.Source_inventory

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   refusal control can plant the defect. [uses] parses only a source spelling the tag. *)
let reading_or_refusal ~fail ~what read =
  match Test_scan.Scan_attempt.run read with
  | Scanned uses -> uses
  | Unparsed why ->
      fail
        (Printf.sprintf
           "%s does not parse as OCaml (%s), so its outside-sweep owners went unchecked" what why);
      []
  | Raised exn ->
      fail
        (Printf.sprintf
           "the outside-sweep owner reader raised %s on %s, which parses -- a defect in the \
            scanner rather than in the source"
           exn what);
      []

(* [sources] maps each relative [.ml] path to its content; [read] looks up any repository file. *)
let run ?exemptions ~fail ~read sources =
  if List.is_empty sources then
    fail "no .ml source is among the files read -- the rule's source tree matches nothing";
  let uses =
    List.concat_map sources ~f:(fun (file, source) ->
        reading_or_refusal ~fail ~what:file (fun () -> Scan.uses ~file source))
  in
  let report = Scan.check ?exemptions ~read uses in
  List.iter report.findings ~f:fail;
  report

(* Each refusal of this file, reached through a synthetic input: the observation is recorded only
   when exactly the one message the format names fired. *)
let refusal_control () =
  let source = "test/operations/sweep_owner_scan.ml" in
  let routed_case label ~format ~fragment run =
    let messages = ref [] in
    run (fun message -> messages := message :: !messages);
    let routed =
      match !messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if routed then Test_scan.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  let read _ = None in
  (* No exemption: the synthetic trees hold no replay probe, and its staleness is the cases'. *)
  let run = run ~exemptions:[] in
  routed_case "an invalid OCaml source spelling the tag reaches the parse refusal"
    ~format:"%s does not parse as OCaml (%s), so its outside-sweep owners went unchecked"
    ~fragment:"t/bad.ml does not parse as OCaml" (fun fail ->
      ignore (run ~fail ~read [ ("t/bad.ml", "let = `Outside_sweep") ] : Scan.report));
  routed_case
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the outside-sweep owner reader raised %s on %s, which parses -- a defect in the scanner \
       rather than in the source"
    ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses" (fun fail ->
      ignore (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted") : _ list));
  routed_case "a run without .ml sources reaches the empty-corpus refusal"
    ~format:"no .ml source is among the files read -- the rule's source tree matches nothing"
    ~fragment:"no .ml source is among the files read" (fun fail ->
      ignore (run ~fail ~read [] : Scan.report));
  Test_scan.Refusal_control_manifest.print source

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--refusal-control" ] -> refusal_control ()
  | _ :: root :: generated ->
      let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
      let read path =
        List.find_map (Inventory.files inventory) ~f:(fun (file : Inventory.file) ->
            Option.some_if (String.equal file.path path) file.on_disk)
        |> Option.map ~f:In_channel.read_all
      in
      let sources =
        Inventory.select inventory ~f:(fun path -> String.is_suffix path ~suffix:".ml")
        |> List.map ~f:(fun (file : Inventory.file) ->
            (file.path, In_channel.read_all file.on_disk))
      in
      let report = run ~fail:Verdict.fail ~read sources in
      List.iter report.resolved ~f:print_endline;
      (* Canaries: a scan gone blind and an intact tree print the same nothing. Each owner kind the
         tree uses today must be read and resolved. *)
      p_exists "the scan resolves the compiler-trace CI step owning cc_backend_trace_name"
        report.resolved ~f:(fun line ->
          String.is_prefix line ~prefix:"test/operations/cc_backend_trace_name.ml: ci-step:ci.yml:");
      p_exists "the scan resolves a plain-run owner of a forced skip" report.resolved
        ~f:(String.is_substring ~substring:": plain-run:");
      Test_scan.Refusal_control_manifest.print "sweep_owner_scan.ml"
  | _ -> Stdlib.exit 2
