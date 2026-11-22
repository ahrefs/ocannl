(* gh-ocannl-2394: turn a saved autotune winner (.sexp from an autotune cache directory) into a
   digest-free replay fixture (Winner_fixture.t).

   Accepted input boundary (fail-closed): reads only Schedule_cache's on-disk stored-entry format
   ((key <k>) (payload <entry>)). Refuses placements- and abandonment- entries, unparseable files,
   non-entry payloads, empty segment lists, and non-positive segment unit counts.

   Usage: dune exec tools/extract_winner_fixture.exe -- <saved-winner.sexp> [-o <fixture.sexp>] *)

open Base
open Stdio

let usage () = eprintf "usage: extract_winner_fixture <saved-winner.sexp> [-o <fixture.sexp>]\n%!"

let parse_args args =
  let rec go input output = function
    | [] -> (
        match input with
        | Some path -> Ok (path, output)
        | None -> Error "missing input <saved-winner.sexp>")
    | ("-h" | "--help") :: _ ->
        usage ();
        Stdlib.exit 0
    | "-o" :: path :: rest -> (
        match output with
        | Some _ -> Error "duplicate -o option"
        | None -> go input (Some path) rest)
    | [ "-o" ] -> Error "missing path after -o"
    | arg :: _ when String.is_prefix arg ~prefix:"-" ->
        Error (Printf.sprintf "unknown option %S" arg)
    | path :: rest -> (
        match input with
        | Some _ -> Error (Printf.sprintf "unexpected extra argument %S" path)
        | None -> go (Some path) output rest)
  in
  go None None args

let () =
  let args = match Array.to_list (Sys.get_argv ()) with _ :: tl -> tl | [] -> [] in
  match parse_args args with
  | Error msg ->
      eprintf "extract_winner_fixture: %s\n%!" msg;
      usage ();
      Stdlib.exit 2
  | Ok (input_path, output_path) -> (
      match Winner_fixture.extract_file input_path with
      | Error msg ->
          eprintf "extract_winner_fixture: %s: %s\n%!" input_path msg;
          Stdlib.exit 1
      | Ok fixture -> (
          match output_path with
          | None ->
              Out_channel.set_binary_mode stdout true;
              Out_channel.output_string stdout (Winner_fixture.to_sexp_string fixture)
          | Some path -> (
              try Winner_fixture.save_file ~path fixture
              with Sys_error msg ->
                eprintf "extract_winner_fixture: cannot write %s: %s\n%!" path msg;
                Stdlib.exit 1)))
