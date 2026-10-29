(* Which backends a dune invocation can hold by name, and whether the configuration counts: the
   reachability half of tools/batch-backends.sh's resolution of a batch's backends (gh-ocannl-1004,
   gh-ocannl-1066, gh-ocannl-1227; the reasoning is Test_utils.Slot_kind's). Run from the repository
   root with the dune argv as its arguments. Prints one `names <backend>: <why>` line per backend a
   reachable stanza's marker names, then a `reads config: <why>` line unless every target is a
   configuration-free alias or a plain compilation product, then `end`; or a single `unknown: <why>`
   line when the argv is unmodelled or the tree cannot be read; or one `refused: <dune file>:<line>:
   <why>` line per backend marker env_var_deps refuses, then `end`, which the runner refuses the
   batch over (MARKER REFUSED) rather than read as every backend. An answer without its `end` line
   is read as unknown too, so a crash cannot pass for a batch that names nothing. *)

open Base
open Stdio

let () =
  let argv = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  match Test_utils.Slot_kind.dune_files ~root:"." () with
  | exception exn -> printf "unknown: the source tree is unreadable here (%s)\n" (Exn.to_string exn)
  | files -> (
      let config_files = Test_utils.Slot_kind.config_files_of ~root:"." files in
      match Test_utils.Slot_kind.answer ~config_files ~dune_files:files argv with
      | Unknown why -> printf "unknown: %s\n" why
      | Refused refused ->
          List.iter refused ~f:(fun why -> printf "refused: %s\n" why);
          printf "end\n"
      | Reaches { named; reads_config } ->
          List.iter named ~f:(fun (backend, why) -> printf "names %s: %s\n" backend why);
          Option.iter reads_config ~f:(fun why -> printf "reads config: %s\n" why);
          printf "end\n")
