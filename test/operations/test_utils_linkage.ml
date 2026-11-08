(* gh-ocannl-2010: a printer or generated-artifact reader must not link the OCaml compiler.
   compiler-libs.common forces its units into every executable on its library dependency path, even
   when no scanner is called. Derive the forbidden population from the installed archive, and
   inspect the units actually linked here, rather than pinning a byte count or module list. This
   checks native linkage of Test_utils and the cheap runtime support libraries declared by its
   stanza; it makes no claim about the size of tests that need a parser. *)

open Base
open Verdict.Claims

let () =
  Test_utils.set_binary_stdout ();
  let tool = Stdlib.Sys.argv.(1) in
  let archive = Stdlib.Sys.argv.(2) in
  let channel = Unix.open_process_args_in tool [| tool; archive |] in
  let text = Stdio.In_channel.input_all channel in
  (match Unix.close_process_in channel with
  | Unix.WEXITED 0 -> ()
  | _ -> fail "ocamlobjinfo could not inventory the compiler archive");
  let compiler_units =
    String.split_lines text
    |> List.filter_map ~f:(fun line ->
        Option.map (String.chop_prefix line ~prefix:"Name: ") ~f:String.strip)
  in
  let linked = Set.of_list (module String) (Dynlink.main_program_units ()) in
  p "the linked-unit census sees the runtime helper library" (Set.mem linked "Test_utils");
  p_none "runtime test helpers link none of the compiler archive's units" compiler_units
    ~f:(fun name ->
      let present = Set.mem linked name in
      if present then Stdio.eprintf "unexpected compiler unit: %s\n%!" name;
      present)
