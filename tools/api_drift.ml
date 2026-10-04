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

let print_side prefix = function
  | None -> ()
  | Some (d : Surface.declaration) ->
      printf "%s %s (line %d)\n" prefix d.name d.line;
      List.iter (String.split_lines d.text) ~f:(fun line -> printf "%s %s\n" prefix line)

let run since until =
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
              List.iter changes ~f:(fun (old, fresh) ->
                  Int.incr total;
                  print_side "-" old;
                  print_side "+" fresh)));
      parent := commit);
  printf "\n%d declaration changes across %d first-parent commits.\n" !total (List.length commits)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; ("--help" | "-h") ] ->
      printf
        "Usage: tools/api-drift.sh <since-rev> [until-rev]\n\
         Reads committed first-parent history (until defaults to HEAD); does not judge \
         compatibility.\n"
  | [ _; since ] -> (
      try run since "HEAD"
      with exn ->
        eprintf "api-drift: %s\n" (Exn.to_string exn);
        Stdlib.exit 1)
  | [ _; since; until ] -> (
      try run since until
      with exn ->
        eprintf "api-drift: %s\n" (Exn.to_string exn);
        Stdlib.exit 1)
  | _ ->
      eprintf "Usage: tools/api-drift.sh <since-rev> [until-rev]\n";
      Stdlib.exit 2
