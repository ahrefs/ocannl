(* gh-ocannl-1309: an executed parity claim about a tensorized routine needs a census judgment
   beside it, since a Tile_mma that declined to the scalar fallback passes parity too. The reader,
   its closed lists of recognised forms and its boundary are in
   test/support/tensorized_census_scan.ml's header. *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Tensorized_census_scan
module Inventory = Test_scan.Source_inventory
module Dune = Test_scan.Dune_stanza_scan
module Sort = Test_scan.Forced_decline_sort

(* A row absorbs every unjudged tensorized parity claim of its file with this label (the label
   expression as the refusal prints it), and says why that claim needs no census. A row nothing
   answers to is stale and fails. The tree needs none today. *)
let exemptions : Scan.exemption list = []

(* gh-ocannl-1308: the forced-decline leg (`Utils.tile_mma_decline_forced`) reruns exactly the
   stanzas declaring this variable, and the population it has to rerun is the one read below: every
   file with a judged executed tensorized claim. So each stanza running such a file's program -- its
   `(test)`, and every rule running its executable, variants included -- must declare it, or the leg
   reads that stanza's cached off-regime result as a surviving census claim. The set is derived
   here, never listed: a stanza is found by the name its `(test)`/`(tests)` gives or by the
   executable an action runs (`Dune_stanza_scan.executables_run`), in the dune file beside the
   source. *)
let hook_var = "OCANNL_TOOL_TEST_FORCE_TILE_MMA_DECLINE"

let undeclared_hook inventory judged_paths =
  List.concat_map judged_paths ~f:(fun path ->
      let dir = Stdlib.Filename.dirname path in
      let name = Stdlib.Filename.(chop_extension (basename path)) in
      let dune = dir ^ "/dune" in
      match
        List.find (Inventory.files inventory) ~f:(fun (f : Inventory.file) ->
            String.equal f.path dune)
      with
      | None -> [ Printf.sprintf "%s judges the census, and no %s runs it" path dune ]
      | Some file -> (
          let content = In_channel.read_all file.on_disk in
          let runs stanza =
            (match Dune.head stanza with
              | Some ("test" | "tests") -> List.mem (Dune.names_of stanza) name ~equal:String.equal
              | _ -> false)
            || List.exists (Dune.executables_run stanza) ~f:(fun (_, command) ->
                match command with
                | Dune.Runs exe | Dune.Runs_dependency exe ->
                    String.equal (Stdlib.Filename.basename exe) (name ^ ".exe")
                | _ -> false)
          in
          let running =
            List.filter (Dune.spelled_stanzas content) ~f:(fun (stanza, _) -> runs stanza)
          in
          match running with
          | [] -> [ Printf.sprintf "%s judges the census, and no stanza of %s runs it" path dune ]
          | _ ->
              List.filter_map running ~f:(fun (stanza, raw) ->
                  Option.some_if
                    (not (Dune.declares_env_var (Dune.field stanza "deps") hook_var))
                    (Printf.sprintf
                       "%s:%d runs %s, which judges the census, and does not declare `(env_var \
                        %s)` -- the forced-decline leg would read its cached result as a surviving \
                        census claim"
                       dune
                       (Dune.line_of content raw.Dune.raw_start)
                       name hook_var))))

let scan ?(pin_harness = true) ?(check_hook = true) ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources =
    Inventory.select inventory ~f:Test_scan.Ll_test_scan.test_source
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
  if check_hook then (
    let judged = Scan.judged_paths sources in
    let undeclared = undeclared_hook inventory judged in
    List.iter undeclared ~f:Verdict.fail;
    p_empty "every stanza running a census-judging test declares the forced-decline variable"
      ~over:judged undeclared);
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

(* [--sort-forced-decline ROOT LOG...]: sort each saved sweep LOG against the census-judging tests
   of the checkout ROOT (its test/ and arrayjit/test/ sources; no build tree is read). Exit 0 when
   every failure is one the forced decline predicts, 1 on an anomaly, 2 on an unreadable log. *)
let sort_logs root logs =
  let sources =
    List.concat_map [ "test"; "arrayjit/test" ] ~f:(fun dir ->
        Inventory.regular_files (Stdlib.Filename.concat root dir))
    |> List.map ~f:(fun on_disk -> (Inventory.relative_to_root ~root on_disk, on_disk))
    (* A [.pp.ml] is dune's marshalled AST, present only when ROOT is a build tree. *)
    |> List.filter ~f:(fun (path, _) ->
        Test_scan.Ll_test_scan.test_source path && not (String.is_suffix path ~suffix:".pp.ml"))
    |> List.map ~f:(fun (path, on_disk) -> (path, In_channel.read_all on_disk))
  in
  let judged = Scan.judged_paths sources in
  if List.is_empty judged then (
    eprintf "no census-judging test under %s/test: is it an OCANNL checkout?\n" root;
    Stdlib.exit 2);
  let anomalous =
    List.count logs ~f:(fun path ->
        match In_channel.read_all path with
        | exception Sys_error e ->
            eprintf "unreadable log: %s\n" e;
            Stdlib.exit 2
        | text ->
            let log = Sort.read text in
            let sorted = Sort.sort ~judged log in
            printf "forced-decline sort of %s:\n%s" path (Sort.render log sorted);
            not (List.is_empty (Sort.anomalies log sorted)))
  in
  Stdlib.exit (if anomalous > 0 then 1 else 0)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: "--sort-forced-decline" :: root :: (_ :: _ as logs) -> sort_logs root logs
  | [ _; "--fixture"; root ] -> scan ~pin_harness:false ~check_hook:false ~exemptions:[] root []
  | [ _; "--fixture-hook"; root ] -> scan ~pin_harness:false ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan ~pin_harness:false ~check_hook:false
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
      Test_scan.Refusal_control_manifest.print "tensorized_census_ratchet.ml"
  | _ -> Stdlib.exit 2
