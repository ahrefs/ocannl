(* Which backends a dune invocation can hold by name, and whether it reads the configuration: the
   reachability half of tools/batch-backends.sh's resolution of a batch's backends (gh-ocannl-1004,
   gh-ocannl-1066, gh-ocannl-1095; the reasoning is Test_utils.Slot_kind's). Run from the repository
   root with the dune argv as its arguments. Prints one `names <backend>: <why>` line per backend a
   reachable stanza's marker names, then a `reads config: <why>` line when a reachable stanza
   selects its backend from the configuration, then `end`; or a single `unknown: <why>` line when
   the argv is unmodelled or the tree cannot be read. An answer without its `end` line is read as
   unknown too, so a crash cannot pass for a batch that names nothing. *)

open Base
open Stdio

(* Every dune file dune itself would read: it skips directories whose name starts with `.` or `_`
   (_build, _opam, .git, ...). Reading more than dune does only widens the answer. *)
let rec dune_files dir =
  let path = if String.is_empty dir then "." else dir in
  let entries = Stdlib.Sys.readdir path |> Array.to_list |> List.sort ~compare:String.compare in
  let here =
    if List.mem entries "dune" ~equal:String.equal then
      [ (dir, In_channel.read_all (Stdlib.Filename.concat path "dune")) ]
    else []
  in
  here
  @ List.concat_map entries ~f:(fun e ->
      let sub = if String.is_empty dir then e else dir ^ "/" ^ e in
      if String.is_prefix e ~prefix:"." || String.is_prefix e ~prefix:"_" then []
      else if Stdlib.Sys.is_directory sub then dune_files sub
      else [])

let () =
  let argv = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  match dune_files "" with
  | exception exn -> printf "unknown: the source tree is unreadable here (%s)\n" (Exn.to_string exn)
  | files -> (
      match Test_utils.Slot_kind.answer ~dune_files:files argv with
      | Unknown why -> printf "unknown: %s\n" why
      | Reaches { named; reads_config } ->
          List.iter named ~f:(fun (backend, why) -> printf "names %s: %s\n" backend why);
          Option.iter reads_config ~f:(fun why -> printf "reads config: %s\n" why);
          printf "end\n")
