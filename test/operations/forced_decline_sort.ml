(* The forced-decline sorter (Test_scan.Forced_decline_sort; the CLI is
   `tensorized_census_ratchet.exe --sort-forced-decline`) over the saved sweeps kept in
   forced_decline_logs/ -- the census-judging stanzas under
   OCANNL_TOOL_TEST_FORCE_TILE_MMA_DECLINE=1 on cc and Metal (lukstafi/ocannl-staging#2233), CUDA
   (rog-nv-linux) and HIP discrete (tuf-amd-linux) (gh-ocannl-2402), each the run record's log with
   its home directory written as `~`.

   The census-judging set is this directory's: the stanzas of those logs all run tests here, and
   [Tensorized_census_scan.judged_paths] judges each file by its own content, so reading this
   directory's sources gives every one of them the verdict the CLI's whole-checkout reading gives.

   The golden pins each sort, reproducing the hand sort: the census failures per stanza across all
   four backends, each backend's expected decline-sensitive claims, the other requested stanzas
   held, no anomaly. Then a planted failure of each anomalous kind, appended to the cc sweep, must
   be flagged with its stanza. Last, the CLI itself runs over all saved logs, rooted at this build
   tree, and its exit statuses are pinned.

   The logs are fixed history; the sources they are sorted against are today's. So renaming a label
   [Forced_decline_sort.decline_sensitive] lists, or changing whether a test judges the census,
   turns a saved sweep's sort red. Do not loosen a glob until it matches both labels: re-capture the
   sweeps (the leg in build-and-test.md#verdict-claims), or, without that hardware, keep the old
   label as a second listed pattern whose reason says it matches the saved sweeps and since when. *)

open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Tensorized_census_scan
module Sort = Test_scan.Forced_decline_sort

let logs_dir = "forced_decline_logs"

let sources =
  Stdlib.Sys.readdir "." |> Array.to_list |> Test_scan.Config_key_scan.sources_among
  (* Dune's preprocessed [.pp.ml] files beside them are marshalled ASTs, not sources. *)
  |> List.filter ~f:(fun f -> not (String.is_suffix f ~suffix:".pp.ml"))
  |> List.sort ~compare:String.compare
  |> List.map ~f:(fun f -> ("test/operations/" ^ f, In_channel.read_all f))

let judged = Scan.judged_paths sources

let sorted_of text =
  let log = Sort.read text in
  (log, Sort.sort ~judged log)

let () =
  p_exists "this directory holds census-judging tests" judged ~f:(fun _ -> true);
  p_empty
    "every decline-sensitive value claim the sorter lists names a claim its source still writes"
    ~over:Sort.decline_sensitive (Sort.stale sources);
  let logs =
    Stdlib.Sys.readdir logs_dir |> Array.to_list
    |> List.filter ~f:(fun f -> String.is_suffix f ~suffix:".txt")
    |> List.sort ~compare:String.compare
  in
  p_exists "the sorter read a saved forced-decline sweep" logs ~f:(fun _ -> true);
  List.iter logs ~f:(fun name ->
      let log, sorted = sorted_of (In_channel.read_all (Stdlib.Filename.concat logs_dir name)) in
      printf "\nforced-decline sort of %s:\n%s" name (Sort.render log sorted);
      p_empty
        (Printf.sprintf "%s: every failure is one the forced decline predicts" name)
        ~over:sorted (Sort.anomalies log sorted))

(* A planted failure of each anomalous kind in the cc sweep -- a block appended to it, or for the
   unfinished run its closing exit line changed: the sort must gain exactly one anomaly, and it must
   name the planted stanza (or the exit status). The anomaly found goes to stderr, so the golden's
   rows are the same whether or not the sorter flags it ([tools/mutant.sh] counts them). *)
let () =
  let text = In_channel.read_all (Stdlib.Filename.concat logs_dir "pr2233_cc.txt") in
  let base_log, base_sorted = sorted_of text in
  (* A red `(test)` stanza of the sweep running a census-judging test. *)
  let red_test =
    List.find_map base_sorted ~f:(fun (s : Sort.sorted_block) ->
        Option.some_if
          (Option.is_some s.source
          && Option.equal String.equal s.block.program (Some s.block.stanza))
          s.block.stanza)
  in
  (* The header dune prints for a failing `(test)`: the stanza's `(name ...)` line, highlighted. *)
  let header name =
    Printf.sprintf "File \"test/operations/dune\", line 1, characters 7-%d:\n1 |  (name %s)\n"
      (7 + String.length name)
      name
  in
  (* The same header as a colorized dune writes it (CLICOLOR_FORCE=1). *)
  let colored_header name =
    Printf.sprintf
      "\027[1mFile \"test/operations/dune\", line 1, characters 7-%d\027[0m:\n\
       \027[34m1 |\027[0m  (name %s)\n"
      (7 + String.length name)
      name
  in
  let append block t = t ^ "\n" ^ block in
  let one_fail label = "FAIL: " ^ label ^ ": false\nFAILED: 1 check did not hold: \"planted\".\n" in
  let planted =
    Option.value_map red_test ~default:[] ~f:(fun stanza ->
        [
          ( "a failed parity claim",
            stanza,
            append
              (header stanza ^ one_fail "planted: the tensorized matmul matches the serial twin") );
          (* The wording of online_softmax_block_mma's gradient claim, which the census scan's
             parity words alone do not read. *)
          ( "a failed tolerance claim",
            stanza,
            append
              (header stanza
              ^ one_fail
                  "planted: a parameter gradient agrees with the composed step within 1e-4 relative"
              ) );
          ( "a failed parity claim under a colorized header",
            stanza,
            append
              (colored_header stanza
              ^ one_fail "planted: the tensorized matmul matches the serial twin") );
          (* Value claims whose labels name neither parity nor the census (qkv_zero_init's and
             tile_mma_narrow's wordings). *)
          ( "a failed value claim in neither vocabulary",
            stanza,
            append
              (header stanza ^ one_fail "planted: fission qkv resets the accumulator on every call")
          );
          ( "a dune error with no location",
            "Alias",
            append "Error: Alias \"nope\" specified on the command line is empty.\n" );
          ( "a crash",
            stanza,
            append (header stanza ^ "Fatal error: exception Failure(\"planted\")\n") );
          ( "an unaccounted failure count",
            stanza,
            append
              (header stanza
             ^ "FAIL: planted structure: false\nFAILED: 2 checks did not hold: \"planted\".\n") );
          ( "a failure outside the census-judging tests",
            "planted_outside",
            append (header "planted_outside" ^ one_fail "planted structure") );
          ( "a golden diff",
            "planted.expected",
            append "File \"test/operations/planted.expected\", line 1, characters 0-0:\n" );
          ( "an unfinished run (a capped batch's exit status)",
            "exit 142",
            fun t ->
              match String.chop_suffix (String.rstrip t) ~suffix:"exit: 1" with
              | Some head -> head ^ "exit: 142\n"
              | None -> t );
        ])
  in
  p_exists "the cc sweep has a red census-judging test stanza to plant in" planted ~f:(fun _ ->
      true);
  let base = Sort.anomalies base_log base_sorted in
  List.iter planted ~f:(fun (kind, needle, plant) ->
      let log, sorted = sorted_of (plant text) in
      let found = Sort.anomalies log sorted in
      let added = List.filter found ~f:(fun a -> not (List.mem base a ~equal:String.equal)) in
      List.iter added ~f:(fun a -> eprintf "planted %s: %s\n" kind a);
      p
        (Printf.sprintf "the sorter flags %s planted in the cc sweep, naming its stanza" kind)
        (List.length found = List.length base + 1
        && List.exists added ~f:(fun a -> String.is_substring a ~substring:needle)));
  (* An unfinished run reports no stanza held: the requested ones that printed nothing may never
     have run. *)
  let unfinished =
    match String.chop_suffix (String.rstrip text) ~suffix:"exit: 1" with
    | Some head -> head ^ "exit: 142\n"
    | None -> text
  in
  let log, sorted = sorted_of unfinished in
  p "an unfinished run's report lists no held stanza"
    ((not (List.is_empty base_log.requested))
    && not (String.is_substring (Sort.render log sorted) ~substring:"  held:"))

(* The CLI, [tensorized_census_ratchet.exe --sort-forced-decline ROOT LOG...], rooted at this build
   tree ([../..], which holds test/ as this test's dependencies put it there): its argv form, its
   whole-checkout source reading and its exit statuses. *)
let () =
  let cli args =
    Stdlib.Sys.command
      (Stdlib.Filename.quote_command
         (Stdlib.Filename.concat Stdlib.Filename.current_dir_name "tensorized_census_ratchet.exe")
         ~stdout:Stdlib.Filename.null ~stderr:Stdlib.Filename.null ("--sort-forced-decline" :: args))
  in
  let saved =
    Stdlib.Sys.readdir logs_dir |> Array.to_list
    |> List.filter ~f:(fun f -> String.is_suffix f ~suffix:".txt")
    |> List.sort ~compare:String.compare
    |> List.map ~f:(Stdlib.Filename.concat logs_dir)
  in
  p "the CLI sorts every saved sweep with no anomaly: exit 0" (cli ("../.." :: saved) = 0);
  let planted = Stdlib.Filename.temp_file "forced_decline_sort" ".txt" in
  Out_channel.write_all planted
    ~data:
      (In_channel.read_all (List.hd_exn saved)
      ^ "\n\
         File \"test/operations/dune\", line 1, characters 7-27:\n\
         1 |  (name schedule_batched_mma)\n\
         FAIL: planted: the tensorized matmul matches the serial twin: false\n\
         FAILED: 1 check did not hold: \"planted\".\n");
  p "the CLI exits 1 on a sweep with a planted parity failure" (cli [ "../.."; planted ] = 1);
  Stdlib.Sys.remove planted;
  p "the CLI exits 2 on an unreadable log" (cli [ "../.."; "no_such_sweep.txt" ] = 2);
  p "the CLI exits 2 on a root holding no census-judging test"
    (cli [ logs_dir; List.hd_exn saved ] = 2)
