(* gh-ocannl-2234: the project's compiler policy refuses undeclared library uses.

   Boundary: read only the top-level [lang] and [implicit_transitive_deps] forms of the supplied
   dune-project, through Dune_stanza_scan. Copy those forms into isolated projects; do not copy the
   workspace's packages, extensions or build rules. Compile real fixtures with the action's Dune,
   without changing the running workspace. The negative fixture must fail specifically on its
   undeclared module, and positive fixtures exercise direct declarations, re-exports and
   Dune-injected ppx runtime libraries. Dropping the setting from the supplied project makes the
   negative fixture compile and this test fail. A permanent default-policy fixture also proves that
   this same undeclared use compiles without the visibility form. Every fixture builds in
   [Fresh_process.nested_dune_environment], and a last control shows what that environment removes:
   a nested Dune handed the action's variables resolves and traces as the parent's action. *)

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

let fixture_root tag =
  let root = Stdlib.Filename.temp_file ("explicit-deps-" ^ tag) "" in
  Unix.unlink root;
  Unix.mkdir root 0o700;
  root

(* A nested Dune is an independent build, not one of the parent's actions. Stdout and stderr come
   back as one log, stdout first; Dune writes its diagnostics to stderr. *)
let nested_check ?cwd ~dune ~env root =
  let status, stdout, stderr =
    Fresh_process.run ~exe:dune ?cwd ~env
      [
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
      ]
  in
  (status, stdout ^ stderr)

let compile ~dune ~project ~tag ~bridge ~libraries ~generated =
  let root = fixture_root tag in
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
  nested_check ~dune ~env:(Fresh_process.nested_dune_environment ()) root

(* Dune copies its own trace into [DUNE_ACTION_TRACE_DIR] from 3.22 on (ocaml/dune#13302); the
   project's floor is older. A version this cannot read leaves the leg open, to be judged. *)
let copies_action_trace ~dune =
  let status, version, _ =
    Fresh_process.run ~exe:dune ~env:(Fresh_process.nested_dune_environment ()) [ "--version" ]
  in
  let number part = Option.try_with (fun () -> Int.of_string part) in
  match (status, String.split (String.strip version) ~on:'.') with
  | Unix.WEXITED 0, major :: minor :: _ -> (
      match (number major, number minor) with
      | Some major, Some minor -> major > 3 || (major = 3 && minor >= 22)
      | _ -> true)
  | _ -> true

(* The control for [Fresh_process.nested_dune_environment]: one failing build per environment, run
   from the fixture's parent directory, each with a planted action-trace directory of its own.
   Handed Dune's action variables, the nested Dune names its root relative to the parent's directory
   ([INSIDE_DUNE]) and copies its trace into the parent's action-trace directory
   ([DUNE_ACTION_TRACE_DIR]); built in the nested environment, it does neither. *)
let environment_control ~dune ~project =
  let trace_leg =
    Verdict.leg ~aggregation:`Environment ~when_:(copies_action_trace ~dune)
      ~on:"a Dune older than 3.22, which copies no action trace" ()
  in
  let root = fixture_root "environment" in
  Out_channel.write_all (root ^ "/dune-project") ~data:project;
  Out_channel.write_all (root ^ "/dune") ~data:"(library (name broken))\n";
  Out_channel.write_all (root ^ "/broken.ml") ~data:"let value = Missing.value\n";
  let relative_entry = "Entering directory '" ^ Stdlib.Filename.basename root ^ "'" in
  let build tag nest =
    let trace_dir = root ^ "/" ^ tag ^ "-trace" in
    Unix.mkdir trace_dir 0o700;
    (* Planted first: Dune, like getenv, takes a name's first binding, so these win over the running
       action's own without relying on the function under test. *)
    let as_action =
      Array.append [| "INSIDE_DUNE=1"; "DUNE_ACTION_TRACE_DIR=" ^ trace_dir |] (Unix.environment ())
    in
    let env = if nest then Fresh_process.nested_dune_environment ~env:as_action () else as_action in
    let status, log = nested_check ~cwd:(Stdlib.Filename.dirname root) ~dune ~env root in
    let ran =
      (not (Poly.equal status (Unix.WEXITED 0)))
      && String.is_substring log ~substring:"Unbound module Missing"
      && String.is_substring log ~substring:"Entering directory '"
    in
    if not ran then eprintf "%s: unexpected nested build (not part of the golden)\n%s" tag log;
    (ran, String.is_substring log ~substring:relative_entry, trace_dir)
  in
  let traces trace_dir = Array.to_list (Stdlib.Sys.readdir trace_dir) in
  let ran, relative, trace_dir = build "as-action" false in
  p "a nested Dune handed the action's variables resolves paths as the parent's action"
    (ran && relative);
  in_leg ~leg:trace_leg
    "a nested Dune handed the action's variables copies its trace into the parent's" (fun label ->
      p_exists label (traces trace_dir) ~f:(String.is_prefix ~prefix:"dune"));
  let ran, relative, trace_dir = build "nested" true in
  p "a nested Dune in the nested-Dune environment resolves paths as an independent build"
    (ran && not relative);
  in_leg ~leg:trace_leg
    "a nested Dune in the nested-Dune environment leaves the parent's action trace alone"
    (fun label -> p_empty label ~over:[ trace_dir ] (traces trace_dir))

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
  environment_control ~dune ~project;
  Test_scan.Refusal_control_manifest.print source
