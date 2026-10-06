open Base
open Stdio
module Surface = Test_utils.Api_drift

let git_raw args =
  let channel = Unix.open_process_args_in "git" (Array.of_list ("git" :: args)) in
  let output = In_channel.input_all channel in
  match Unix.close_process_in channel with
  | Unix.WEXITED 0 -> output
  | _ -> failwith ("git failed: " ^ String.concat ~sep:" " args)

let git args = String.strip (git_raw args)
let lines text = if String.is_empty text then [] else String.split_lines text
let paths rev = git [ "ls-tree"; "-r"; "--name-only"; rev ] |> lines

let source_paths rev =
  let paths = paths rev in
  let dunes =
    List.filter paths ~f:(fun p ->
        String.equal (Stdlib.Filename.basename p) "dune"
        && Test_utils.Dead_export_scan.in_scan_root p)
    |> List.map ~f:(fun p -> (p, git_raw [ "show"; rev ^ ":" ^ p ]))
  in
  Surface.sources ~dunes paths

let resolve rev = git [ "rev-parse"; "--verify"; "--end-of-options"; rev ^ "^{commit}" ]

let read rev source =
  let inventory = if String.equal (Stdlib.Filename.basename source) "dune" then paths rev else [] in
  git_raw [ "show"; rev ^ ":" ^ source ] |> Surface.declarations ~paths:inventory ~source

let run ?context since until =
  let since = resolve since and until = resolve until in
  let history = git [ "rev-list"; "--first-parent"; until ] |> lines in
  if not (List.mem history since ~equal:String.equal) then
    failwith "since-rev must be a commit on until-rev's first-parent history";
  let commits = git [ "rev-list"; "--first-parent"; "--reverse"; since ^ ".." ^ until ] |> lines in
  printf "Public source declarations: %s..%s\n" since until;
  printf
    "Editorial aid; .ml bodies flag possible inferred-type drift. PPX exports and inferred types \
     need manual review. Generator input entries require review of the generated interface. Dune \
     configuration entries retain literal inputs; availability and declaration effects require \
     manual review. Ordinary library dependencies are excluded.\n";
  Option.iter context ~f:(fun context ->
      printf
        "Compact rendering: an entry changed on both sides prints its changed lines (-/+) with %d \
         unchanged lines of context (two-space indent) and counts the rest on ~ lines. Full \
         declaration text: tools/api-drift.sh %s %s\n"
        context since until);
  let parent = ref since and total = ref 0 in
  List.iter commits ~f:(fun commit ->
      let before = source_paths !parent |> Set.of_list (module String) in
      let after = source_paths commit |> Set.of_list (module String) in
      let changed =
        git [ "diff"; "--name-only"; !parent; commit; "--" ] |> lines |> Set.of_list (module String)
      in
      let membership_changes = Set.union (Set.diff before after) (Set.diff after before) in
      let candidates = Set.union changed membership_changes in
      let reported = ref false in
      Set.iter candidates ~f:(fun source ->
          if Set.mem before source || Set.mem after source then
            let old = if Set.mem before source then read !parent source else [] in
            let fresh = if Set.mem after source then read commit source else [] in
            let changes = Surface.changes old fresh in
            if not (List.is_empty changes) then (
              if not !reported then (
                printf "\ncommit %s %s\n" commit (git [ "show"; "-s"; "--format=%s"; commit ]);
                reported := true);
              printf "\n%s\n" source;
              List.iter changes ~f:(fun change ->
                  Int.incr total;
                  List.iter (Surface.render ?context change) ~f:print_endline)));
      parent := commit);
  printf "\n%d declaration changes across %d first-parent commits.\n" !total (List.length commits)

let usage = "Usage: tools/api-drift.sh [--context N] <since-rev> [until-rev]"

let () =
  let fail message =
    eprintf "%s\n" message;
    Stdlib.exit 2
  in
  match List.tl_exn (Array.to_list Stdlib.Sys.argv) with
  | [ ("--help" | "-h") ] ->
      printf
        "%s\n\
         Reads committed first-parent history (until defaults to HEAD); does not judge \
         compatibility. --context N prints only the changed lines of an entry changed on both \
         sides, with N unchanged lines around each; without it, both sides print in full.\n"
        usage
  | args -> (
      let context, positional =
        match args with
        | "--context" :: count :: rest -> (
            match Int.of_string_opt count with
            | Some n when n >= 0 -> (Some n, rest)
            | _ -> fail ("api-drift: --context needs a non-negative line count\n" ^ usage))
        | rest -> (None, rest)
      in
      let since, until =
        match positional with
        | [ since ] -> (since, "HEAD")
        | [ since; until ] -> (since, until)
        | _ -> fail usage
      in
      try run ?context since until
      with exn ->
        eprintf "api-drift: %s\n" (Exn.to_string exn);
        Stdlib.exit 1)
