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

let () =
  let argv = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  match Test_utils.Slot_kind.dune_files ~root:"." () with
  | exception exn -> printf "unknown: the source tree is unreadable here (%s)\n" (Exn.to_string exn)
  | files -> (
      match Test_utils.Slot_kind.answer ~dune_files:files argv with
      | Unknown why -> printf "unknown: %s\n" why
      | Reaches { named; reads_config } ->
          List.iter named ~f:(fun (backend, why) -> printf "names %s: %s\n" backend why);
          Option.iter reads_config ~f:(fun why -> printf "reads config: %s\n" why);
          printf "end\n")
