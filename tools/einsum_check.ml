(* Parse einsum / axis-label specs the way the OCANNL PPXs read a literal spec: as an einsum spec
   first, falling back to an axis-labels spec. Prints each spec's reading on stdout and the parse
   error on stderr; exits 1 if any spec fails.

   Usage: dune exec tools/einsum_check.exe -- '..batch.., seq | heads; ... => ...' 'ab' *)

open Base
open Stdio

let check spec =
  match Einsum_parser.einsum_of_spec spec with
  | rhses, _ ->
      printf "ok: einsum spec with %d RHS(es): %s\n" (List.length rhses) spec;
      true
  | exception Einsum_parser.Parse_error _ -> (
      match Einsum_parser.axis_labels_of_spec spec with
      | _ ->
          printf "ok: axis labels spec: %s\n" spec;
          true
      | exception Einsum_parser.Parse_error msg ->
          eprintf "error: %s\n" msg;
          false)

let () =
  match List.tl (Array.to_list (Sys.get_argv ())) with
  | None | Some [] ->
      eprintf "usage: einsum_check SPEC...\n";
      Stdlib.exit 2
  | Some specs ->
      let results = List.map specs ~f:check in
      if not (List.for_all results ~f:Fn.id) then Stdlib.exit 1
