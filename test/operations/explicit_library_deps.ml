(* gh-ocannl-2234: the project's compiler policy refuses undeclared library uses.

   Boundary: read only the top-level [lang] and [implicit_transitive_deps] forms of the supplied
   dune-project, through Dune_stanza_scan. Copy those forms into isolated projects; do not copy the
   workspace's packages, extensions or build rules. Compile real fixtures with the action's Dune,
   without changing the running workspace. The negative fixture must fail specifically on its
   undeclared module, and positive fixtures exercise direct declarations, re-exports and
   Dune-injected ppx runtime libraries. Dropping the setting from the supplied project makes the
   negative fixture compile and this test fail. A permanent default-policy fixture also proves that
   this same undeclared use compiles without the visibility form. *)

open Base
open Stdio
open Verdict.Claims

let source = "explicit_library_deps.ml"

let project_settings path =
  let settings =
    Test_scan.Dune_stanza_scan.stanzas (In_channel.read_all path)
    |> List.filter ~f:(function
      | Sexp.List (Sexp.Atom ("lang" | "implicit_transitive_deps") :: _) -> true
      | _ -> false)
  in
  let render forms = List.map forms ~f:Sexp.to_string |> String.concat ~sep:"\n" in
  let default_settings =
    List.filter settings ~f:(function
      | Sexp.List (Sexp.Atom "implicit_transitive_deps" :: _) -> false
      | _ -> true)
  in
  (render settings, render default_settings)

let compile ~dune ~project ~tag ~bridge ~libraries ~generated =
  let root = Stdlib.Filename.temp_file ("explicit-deps-" ^ tag) "" in
  Unix.unlink root;
  Unix.mkdir root 0o700;
  let write name data = Out_channel.write_all (root ^ "/" ^ name) ~data in
  write "dune-project" project;
  write "owned.ml" "let value = 7\n";
  write "bridge.ml" "let value = Owned.value\n";
  write "consumer.ml"
    (if generated then "let value ~loc = [%expr 7]\n" else "let value = Owned.value\n");
  write "dune"
    (Printf.sprintf
       "(library (name owned) (modules owned))\n\
        (library (name bridge) (modules bridge) (libraries %s))\n\
        (library (name consumer) (modules consumer) %s %s)\n"
       bridge libraries
       (if generated then "(preprocess (pps ppxlib.metaquot))" else ""));
  let log = root ^ "/compiler.log" in
  let fd = Unix.openfile log [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
  (* A nested Dune is an independent build, not an action sharing the parent's lock. *)
  let env =
    Unix.environment ()
    |> Array.filter ~f:(fun kv ->
        not
          (List.exists [ "INSIDE_DUNE="; "DUNE_ACTION_TRACE_DIR=" ] ~f:(fun prefix ->
               String.is_prefix kv ~prefix)))
  in
  let pid =
    Exn.protect
      ~finally:(fun () -> Unix.close fd)
      ~f:(fun () ->
        Unix.create_process_env dune
          [|
            dune;
            "build";
            "--root";
            root;
            "--build-dir";
            root ^ "/_build";
            "--display";
            "quiet";
            "@check";
            "-j";
            "1";
          |]
          env Unix.stdin fd fd)
  in
  let _, status = Unix.waitpid [] pid in
  (status, In_channel.read_all log)

let () =
  let dune, (project, default_project) =
    match Stdlib.Sys.argv with
    | [| _; dune; project |] -> (dune, project_settings project)
    | _ -> failwith "usage: explicit_library_deps DUNE DUNE_PROJECT"
  in
  let run project tag bridge libraries generated =
    compile ~dune ~project ~tag ~bridge ~libraries ~generated
  in
  let accepted project tag bridge libraries generated =
    let status, log = run project tag bridge libraries generated in
    match status with
    | WEXITED 0 -> true
    | _ ->
        eprintf "%s: unexpected compiler failure (not part of the golden)\n%s" tag log;
        false
  in
  p "a direct library declaration compiles"
    (accepted project "direct" "owned" "(libraries bridge owned)" false);
  p "without the project setting, the same undeclared use compiles"
    (accepted default_project "default-policy" "owned" "(libraries bridge)" false);
  let status, log = run project "undeclared" "owned" "(libraries bridge)" false in
  p "an undeclared transitive library is refused for its unbound module"
    ((not (Poly.equal status (Unix.WEXITED 0)))
    && String.is_substring log ~substring:"Unbound module Owned");
  p "a declared library's re-export compiles"
    (accepted project "re-export" "(re_export owned)" "(libraries bridge)" false);
  p "Dune's injected ppx runtime dependency compiles without a declaration"
    (accepted project "ppx-runtime" "owned" "" true);
  Test_scan.Refusal_control_manifest.print source
