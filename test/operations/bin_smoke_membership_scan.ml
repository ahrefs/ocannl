(** gh-ocannl-874: every public executable declared in [bin/dune] is run exactly once by
    [@bin-smoke].

    The public declarations and the smoke action are independent lists in one dune file. A new
    executable therefore compiles under [@check] even when its runtime canary is accidentally
    omitted. This scan relates those two lists from the parsed dune structure. It accepts either the
    executable's local [%{exe:...}] identity or its [%{bin:...}] public identity, and requires exact
    membership so a stale or duplicated smoke command is visible too. Every repository dune file is
    inspected because [@bin-smoke] is recursive from the workspace root. Conditional executable or
    smoke stanzas, opaque smoke actions, and unexpanded [(include ...)] forms are refused: without
    evaluating or expanding them, a syntactic census could claim commands CI does not execute or
    miss additional commands that it does. The one exception is a platform-gated probe outside
    [bin/], pinned by dune file, program and condition in [platform_contributions] below and exempt
    only while it is exactly that.

    [--negative-control] applies the production scan to a synthetic dune file with two public
    executables and only one smoke command. Its dune rule accepts exit 1 and no other status, so a
    checker that stops detecting the omission makes the control fail rather than go green. *)

open Base
open Stdio
open Verdict.Claims
module Dune_scan = Test_utils.Dune_stanza_scan

type declaration = { local : string; public : string }
type smoke_target = Local of string | Public of string

type result = {
  declarations : declaration list;
  smoked : string list;
  missing : string list;
  unexpected : string list;
  errors : string list;
}

type alias_node = {
  id : string;
  dune_path : string;
  subdir : string;
  stanza : Sexp.t;
  aliases : string list;
  is_bin_smoke : bool;
  dependencies : string list;
  target_dependencies : string list;
  dependency_errors : string list;
}

let sorted = List.sort ~compare:String.compare
let verified_helper_alias = "bin/bin-smoke-env_spelling_gate"
let verified_helper_local = "bin/env_spelling_gate.exe"
let verified_helper_command = "%{dep:env_spelling_gate.exe}"

(* The contributions to [@bin-smoke] that are neither a public bin executable nor unconditional: a
   standalone probe for hardware only some machines have, gated by the platform predicate that
   decides whether it can run at all (gh-ocannl-905). Pinned the way the environment gate's command
   is pinned above -- by dune file, by executable path, and by the exact condition -- because the
   two refusals such a stanza otherwise draws are load-bearing everywhere else.

   Neither claim this scan makes weakens under it. The census is of bin/'s PUBLIC executables, and a
   probe outside bin/ is none of them; the conditionality it announces is about the probe alone, and
   the exemption is spent only where the stanza runs that one program, so a public executable whose
   only smoke run is platform-gated is still reported as missing and any second program in the
   stanza takes the whole exemption away. *)
type platform_contribution = {
  contribution_dune : string;
  contribution_local : string;
  contribution_condition : Sexp.t list;
}

let platform_contributions =
  [
    {
      contribution_dune = "benchmarks/runners/ocannl/dune";
      contribution_local = "benchmarks/runners/ocannl/metal_queue_probe.exe";
      contribution_condition =
        [ Sexp.List [ Sexp.Atom "="; Sexp.Atom "%{system}"; Sexp.Atom "macosx" ] ];
    };
  ]

let path_dirname path =
  match String.rsplit2 path ~on:'/' with Some (directory, _) -> directory | None -> ""

let path_basename path =
  match String.rsplit2 path ~on:'/' with Some (_, basename) -> basename | None -> path

let duplicates strings =
  List.sort_and_group strings ~compare:String.compare
  |> List.filter_map ~f:(function first :: _ :: _ -> Some first | _ -> None)

let aliases_of stanza =
  match Dune_scan.head stanza with
  | Some "alias" -> (
      match Dune_scan.field stanza "name" with Some [ Sexp.Atom name ] -> [ name ] | _ -> [])
  | _ ->
      List.concat_map [ "alias"; "aliases" ] ~f:(fun field ->
          match Dune_scan.field stanza field with
          | None -> []
          | Some args ->
              List.filter_map args ~f:(function Sexp.Atom name -> Some name | _ -> None))

let has_enabled_if stanza = Option.is_some (Dune_scan.field stanza "enabled_if")

(* The pinned contribution a stanza IS, if any: the same dune file and the same condition, spelled
   exactly. A stanza at that path without the condition is an ordinary unconditional contribution
   and gets no exemption; one carrying a different condition is a different claim about when it
   runs, and is refused like any other. *)
let platform_contribution_of ~dune_path stanza =
  List.find platform_contributions ~f:(fun contribution ->
      String.equal contribution.contribution_dune dune_path
      &&
      match Dune_scan.field stanza "enabled_if" with
      | Some condition -> List.equal Sexp.equal condition contribution.contribution_condition
      | None -> false)

let alias_key ~subdir name = Dune_scan.normalize_path (Dune_scan.in_subdir subdir name)

let expanding_dependency_pform atom =
  List.exists (Dune_scan.pieces atom) ~f:(function
    | Dune_scan.Pform pform -> (
        match String.lsplit2 pform ~on:':' with
        | Some (prefix, _) ->
            List.mem [ "read"; "read-lines"; "read-strings" ] prefix ~equal:String.equal
        | None -> false)
    | Dune_scan.Literal _ -> false)

let unmodeled_dependency_pform atom =
  List.exists (Dune_scan.pieces atom) ~f:(function
    | Dune_scan.Pform pform -> (
        match String.lsplit2 pform ~on:':' with
        | Some (prefix, _) ->
            not
              (List.mem
                 [ "dep"; "file"; "path"; "read"; "read-lines"; "read-strings" ]
                 prefix ~equal:String.equal)
        | None -> true)
    | Dune_scan.Literal _ -> false)

let alias_dependencies_in_field ~field ~subdir stanza =
  let rec collect = function
    | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom name ] -> ([ alias_key ~subdir name ], [])
    | Sexp.List (Sexp.Atom "include" :: _) ->
        ([], [ "@bin-smoke reaches an included dependency specification that cannot be expanded" ])
    | Sexp.List (Sexp.Atom "alias_rec" :: _) ->
        ([], [ "@bin-smoke reaches alias_rec, whose recursive dependency cannot be derived" ])
    | Sexp.List children ->
        List.fold children ~init:([], []) ~f:(fun (dependencies, errors) child ->
            let found, found_errors = collect child in
            (List.rev_append found dependencies, List.rev_append found_errors errors))
    | Sexp.Atom atom ->
        if expanding_dependency_pform atom then
          ([], [ "@bin-smoke reaches a dependency specification expanded from file contents" ])
        else if unmodeled_dependency_pform atom then
          ([], [ "@bin-smoke reaches a dependency path computed by an unmodeled pform" ])
        else ([], [])
  in
  match Dune_scan.field stanza field with None -> ([], []) | Some deps -> collect (Sexp.List deps)

let alias_dependencies ~subdir stanza = alias_dependencies_in_field ~field:"deps" ~subdir stanza
let dependency_pforms = [ "dep"; "file"; "path"; "read"; "read-lines"; "read-strings" ]

let dependency_pform_path ~subdir pform =
  match String.lsplit2 pform ~on:':' with
  | Some (prefix, path) when List.mem dependency_pforms prefix ~equal:String.equal ->
      Some (Dune_scan.normalize_path (Dune_scan.in_subdir subdir path))
  | _ -> None

let dependency_paths ~subdir atom =
  match Dune_scan.pieces atom with
  | [ Dune_scan.Literal path ] -> [ Dune_scan.normalize_path (Dune_scan.in_subdir subdir path) ]
  | pieces ->
      List.filter_map pieces ~f:(function
        | Dune_scan.Pform pform -> dependency_pform_path ~subdir pform
        | Dune_scan.Literal _ -> None)

let literal_dependency_path ~subdir atom =
  match Dune_scan.pieces atom with
  | [ Dune_scan.Literal path ] -> Some (Dune_scan.normalize_path (Dune_scan.in_subdir subdir path))
  | [ Dune_scan.Pform _ ] | _ -> None

let target_dependencies_in_field ~field ~subdir stanza =
  let rec collect = function
    | Sexp.Atom atom -> dependency_paths ~subdir atom
    | Sexp.List (Sexp.Atom ("alias" | "alias_rec" | "env_var" | "include" | "universe") :: _) -> []
    | Sexp.List (Sexp.Atom name :: values) when String.is_prefix name ~prefix:":" ->
        List.concat_map values ~f:collect
    | Sexp.List children -> List.concat_map children ~f:collect
  in
  match Dune_scan.field stanza field with
  | None -> []
  | Some deps -> List.concat_map deps ~f:collect

let target_dependencies ~subdir stanza =
  let field_dependencies = target_dependencies_in_field ~field:"deps" ~subdir stanza in
  let action_dependencies =
    match Dune_scan.field stanza "action" with
    | None -> []
    | Some action ->
        List.concat_map action ~f:Dune_scan.atoms
        |> List.concat_map ~f:(fun atom ->
            List.filter_map (Dune_scan.pieces atom) ~f:(function
              | Dune_scan.Pform pform -> dependency_pform_path ~subdir pform
              | Dune_scan.Literal _ -> None))
  in
  let rec literal_action_dependencies ~subdir = function
    | Sexp.Atom _ -> []
    | Sexp.List (Sexp.Atom "no-infer" :: _) -> []
    | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: nested)
      when not (String.is_substring dir ~substring:"%{") ->
        List.concat_map nested
          ~f:(literal_action_dependencies ~subdir:(Dune_scan.in_subdir subdir dir))
    | Sexp.List (Sexp.Atom "with-stdin-from" :: Sexp.Atom input :: nested) ->
        Option.to_list (literal_dependency_path ~subdir input)
        @ List.concat_map nested ~f:(literal_action_dependencies ~subdir)
    | Sexp.List (Sexp.Atom "cat" :: inputs) ->
        List.filter_map inputs ~f:(function
          | Sexp.Atom input -> literal_dependency_path ~subdir input
          | Sexp.List _ -> None)
    | Sexp.List
        (Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive" | "format-dune-file")
        :: Sexp.Atom input
        :: _target :: _) ->
        Option.to_list (literal_dependency_path ~subdir input)
    | Sexp.List (Sexp.Atom ("diff" | "diff?" | "cmp") :: Sexp.Atom left :: Sexp.Atom right :: _) ->
        List.filter_map [ left; right ] ~f:(literal_dependency_path ~subdir)
    | Sexp.List children -> List.concat_map children ~f:(literal_action_dependencies ~subdir)
  in
  let literal_action_dependencies =
    match Dune_scan.field stanza "action" with
    | None -> []
    | Some action -> literal_action_dependencies ~subdir (Sexp.List action)
  in
  List.rev_append field_dependencies
    (List.rev_append action_dependencies literal_action_dependencies)
  |> List.dedup_and_sort ~compare:String.compare

let resolve_target ~subdir target = Dune_scan.normalize_path (Dune_scan.in_subdir subdir target)

let rec inferred_action_targets ~subdir = function
  | Sexp.Atom _ -> []
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> []
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: _) when String.is_substring dir ~substring:"%{"
    ->
      []
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: nested)
    when not (String.is_substring dir ~substring:"%{") ->
      List.concat_map nested ~f:(inferred_action_targets ~subdir:(Dune_scan.in_subdir subdir dir))
  | Sexp.List
      (Sexp.Atom ("with-stdout-to" | "with-stderr-to" | "with-outputs-to")
      :: Sexp.Atom target
      :: nested) ->
      resolve_target ~subdir target :: List.concat_map nested ~f:(inferred_action_targets ~subdir)
  | Sexp.List (Sexp.Atom ("write-file" | "touch" | "mkdir") :: Sexp.Atom target :: _) ->
      [ resolve_target ~subdir target ]
  | Sexp.List
      (Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive" | "format-dune-file")
      :: _source
      :: Sexp.Atom target
      :: _) ->
      [ resolve_target ~subdir target ]
  | Sexp.List children -> List.concat_map children ~f:(inferred_action_targets ~subdir)

let rec inferred_action_directory_targets ~subdir = function
  | Sexp.Atom _ -> []
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> []
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: _) when String.is_substring dir ~substring:"%{"
    ->
      []
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: nested) ->
      List.concat_map nested
        ~f:(inferred_action_directory_targets ~subdir:(Dune_scan.in_subdir subdir dir))
  | Sexp.List (Sexp.Atom "mkdir" :: Sexp.Atom target :: _) -> [ resolve_target ~subdir target ]
  | Sexp.List children -> List.concat_map children ~f:(inferred_action_directory_targets ~subdir)

let produced_targets ~subdir stanza =
  match Dune_scan.head stanza with
  | Some "rule" ->
      let declared =
        List.concat_map [ "target"; "targets" ] ~f:(fun field ->
            match Dune_scan.field stanza field with
            | None -> []
            | Some targets ->
                List.filter_map targets ~f:(function
                  | Sexp.Atom target -> Some (resolve_target ~subdir target)
                  | Sexp.List [ Sexp.Atom "dir"; Sexp.Atom target ] ->
                      Some (resolve_target ~subdir target)
                  | Sexp.List _ -> None))
      in
      let inferred =
        match Dune_scan.field stanza "action" with
        | None -> []
        | Some action -> inferred_action_targets ~subdir (Sexp.List action)
      in
      List.rev_append declared inferred |> List.dedup_and_sort ~compare:String.compare
  | _ -> []

let produced_directory_targets ~subdir stanza =
  match Dune_scan.head stanza with
  | Some "rule" ->
      let declared =
        List.concat_map [ "target"; "targets" ] ~f:(fun field ->
            match Dune_scan.field stanza field with
            | None -> []
            | Some targets ->
                List.filter_map targets ~f:(function
                  | Sexp.List [ Sexp.Atom "dir"; Sexp.Atom target ] ->
                      Some (resolve_target ~subdir target)
                  | Sexp.Atom _ | Sexp.List _ -> None))
      in
      let inferred =
        match Dune_scan.field stanza "action" with
        | None -> []
        | Some action -> inferred_action_directory_targets ~subdir (Sexp.List action)
      in
      List.rev_append declared inferred |> List.dedup_and_sort ~compare:String.compare
  | _ -> []

let rec contains_head expected = function
  | Sexp.Atom _ -> false
  | Sexp.List (Sexp.Atom head :: children) ->
      String.equal head expected || List.exists children ~f:(contains_head expected)
  | Sexp.List children -> List.exists children ~f:(contains_head expected)

let accepts_nonstandard_exit stanza =
  match Dune_scan.field stanza "action" with
  | None -> false
  | Some action -> contains_head "with-accepted-exit-codes" (Sexp.List action)

let public_condition_error names =
  Printf.sprintf
    "public executable `%s` uses enabled_if, so its active membership cannot be derived statically"
    (String.concat ~sep:", " names)

let public_action_preprocess_error names =
  Printf.sprintf
    "public executable `%s` uses an action preprocessor, so its build-time executions are opaque"
    (String.concat ~sep:", " names)

let library_action_preprocess_error dune_path =
  Printf.sprintf
    "%s defines a library action preprocessor, so public executable build-time executions are \
     opaque"
    dune_path

let has_action_preprocessor stanza =
  match Dune_scan.field stanza "preprocess" with
  | Some preprocess -> contains_head "action" (Sexp.List preprocess)
  | None -> false

let smoke_condition_error dune_path =
  Printf.sprintf
    "%s: @bin-smoke uses enabled_if, so its commands are not unconditional runtime coverage"
    dune_path

let include_error dune_path =
  Printf.sprintf "%s uses an unexpanded include stanza, so @bin-smoke membership may be incomplete"
    dune_path

let data_only_dirs_error dune_path =
  Printf.sprintf
    "%s uses data_only_dirs, so the active repository Dune-file census cannot be derived" dune_path

let unresolved_target_chdir_error dune_path =
  Printf.sprintf "%s infers action targets below a chdir whose destination cannot be resolved"
    dune_path

let unresolved_inferred_target_error dune_path =
  Printf.sprintf "%s infers an action target whose path contains a pform" dune_path

let bin_install_error dune_path =
  Printf.sprintf "%s installs files into section bin outside public executable declarations"
    dune_path

let generated_public_run_error dune_path public targets =
  Printf.sprintf "%s generates %s while running public executable %s" dune_path
    (String.concat ~sep:", " targets) public

let target_producer_command_error dune_path error =
  Printf.sprintf "%s: target-producing rule contains a command opaque to the census: %s" dune_path
    error

let target_producer_dependency_error dune_path target =
  let target = match target with Local path -> path | Public name -> "%{bin:" ^ name ^ "}" in
  Printf.sprintf "%s: target-producing rule depends on an alias that runs public executable %s"
    dune_path target

let private_producer_run_error dune_path target =
  let target = match target with Local path -> path | Public name -> "%{bin:" ^ name ^ "}" in
  Printf.sprintf "%s: target-producing graph runs non-public workspace executable %s" dune_path
    target

let target_producer_dependency_spec_error dune_path error =
  Printf.sprintf "%s: target-producing dependency graph is opaque: %s" dune_path error

let executable_dependency_error dune_path target =
  let target = match target with Local path -> path | Public name -> "%{bin:" ^ name ^ "}" in
  Printf.sprintf "%s: public executable build dependency runs public executable %s" dune_path target

let executable_dependency_spec_error dune_path error =
  Printf.sprintf "%s: public executable build dependency graph is opaque: %s" dune_path error

let absolute_chdir_error dune_path cwd =
  Printf.sprintf "%s: @bin-smoke runs a command beneath absolute chdir %s" dune_path cwd

let installs_into_bin stanza =
  match (Dune_scan.head stanza, Dune_scan.field stanza "section") with
  | Some "install", Some [ Sexp.Atom "bin" ] -> true
  | _ -> false

let rec action_infers_target = function
  | Sexp.Atom _ -> false
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> false
  | Sexp.List (Sexp.Atom ("with-stdout-to" | "with-stderr-to" | "with-outputs-to") :: _)
  | Sexp.List (Sexp.Atom ("write-file" | "touch" | "mkdir") :: _)
  | Sexp.List
      (Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive" | "format-dune-file") :: _) ->
      true
  | Sexp.List children -> List.exists children ~f:action_infers_target

let rec has_unresolved_target_chdir = function
  | Sexp.Atom _ -> false
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> false
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: nested)
    when String.is_substring dir ~substring:"%{" ->
      List.exists nested ~f:action_infers_target
      || List.exists nested ~f:has_unresolved_target_chdir
  | Sexp.List children -> List.exists children ~f:has_unresolved_target_chdir

let target_has_pform = function
  | Sexp.Atom target -> String.is_substring target ~substring:"%{"
  | Sexp.List _ -> false

let rec has_unresolved_inferred_target = function
  | Sexp.Atom _ -> false
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> false
  | Sexp.List (Sexp.Atom ("with-stdout-to" | "with-stderr-to" | "with-outputs-to") :: target :: _)
  | Sexp.List (Sexp.Atom ("write-file" | "touch" | "mkdir") :: target :: _) ->
      target_has_pform target
  | Sexp.List
      (Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive" | "format-dune-file")
      :: _source :: target :: _) ->
      target_has_pform target
  | Sexp.List children -> List.exists children ~f:has_unresolved_inferred_target

let unresolved_target_chdir stanza =
  match Dune_scan.head stanza with
  | Some "rule"
    when Option.is_none (Dune_scan.field stanza "target")
         && Option.is_none (Dune_scan.field stanza "targets") -> (
      match Dune_scan.field stanza "action" with
      | Some action -> has_unresolved_target_chdir (Sexp.List action)
      | None -> false)
  | _ -> false

let unresolved_inferred_target stanza =
  match Dune_scan.head stanza with
  | Some "rule"
    when Option.is_none (Dune_scan.field stanza "target")
         && Option.is_none (Dune_scan.field stanza "targets") -> (
      match Dune_scan.field stanza "action" with
      | Some action -> has_unresolved_inferred_target (Sexp.List action)
      | None -> false)
  | _ -> false

let opaque_smoke_error dune_path command =
  Printf.sprintf "%s: @bin-smoke contains an opaque command: %s" dune_path command

let absolute_smoke_error dune_path path =
  Printf.sprintf "%s: @bin-smoke runs an absolute executable path: %s" dune_path path

let bare_smoke_error dune_path path =
  Printf.sprintf "%s: @bin-smoke runs a bare executable name through PATH: %s" dune_path path

let external_smoke_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches an external command whose effects are opaque" dune_path

let accepted_exit_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches with-accepted-exit-codes, so failure is not a canary"
    dune_path

let dynamic_run_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches dynamic-run, whose execution count is not fixed" dune_path

let directory_path_error dune_path =
  Printf.sprintf
    "%s overrides command resolution for a directory containing an @bin-smoke definition" dune_path

let implicit_alias_error dune_path dependency =
  Printf.sprintf "%s: @bin-smoke reaches implicit alias %s, whose contributors are not explicit"
    dune_path dependency

let implicit_runner_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches a test stanza whose default runner is implicit" dune_path

let may_have_implicit_contributors dependency =
  let name = path_basename dependency in
  List.mem
    [ "all"; "check"; "default"; "doc"; "doc-private"; "fmt"; "install"; "lint"; "runtest" ]
    name ~equal:String.equal
  || String.is_prefix name ~prefix:"runtest-"

let cached_alias_error dune_path targets =
  Printf.sprintf "%s: @bin-smoke reaches a target-bearing rule that may be cached: %s" dune_path
    (String.concat ~sep:", " targets)

let cached_command_alias_error dune_path aliases =
  Printf.sprintf
    "%s: @bin-smoke reaches command-bearing alias %s without a direct (universe) dependency"
    dune_path (String.concat ~sep:", " aliases)

let depends_on_universe stanza =
  let rec contains = function
    | Sexp.List [ Sexp.Atom "universe" ] -> true
    | Sexp.List children -> List.exists children ~f:contains
    | Sexp.Atom _ -> false
  in
  match Dune_scan.field stanza "deps" with
  | None -> false
  | Some dependencies -> contains (Sexp.List dependencies)

let declarations_of_stanza ~subdir stanza =
  match Dune_scan.head stanza with
  | Some ("executable" | "executables") ->
      let names = Dune_scan.names_of stanza in
      let public_names = Dune_scan.public_names stanza in
      let has_public_name = List.exists public_names ~f:(Fn.non (String.equal "-")) in
      let condition_errors =
        if has_public_name && has_enabled_if stanza then [ public_condition_error names ] else []
      in
      let preprocess_errors =
        if has_public_name && has_action_preprocessor stanza then
          [ public_action_preprocess_error names ]
        else []
      in
      let declaration_errors = List.rev_append preprocess_errors condition_errors in
      if List.is_empty public_names then ([], [])
      else if List.length names <> List.length public_names then
        ( [],
          declaration_errors
          @ [
              Printf.sprintf "public executable stanza has %d local name(s) but %d public name(s)"
                (List.length names) (List.length public_names);
            ] )
      else
        ( List.zip_exn names public_names
          |> List.filter_map ~f:(fun (name, public) ->
              if String.equal public "-" then None
              else
                Some
                  {
                    local = Dune_scan.normalize_path (Dune_scan.in_subdir subdir (name ^ ".exe"));
                    public;
                  }),
          declaration_errors )
  | _ -> ([], [])

let smoke_targets_of_stanza ~allow_verified_helper ~dune_path ~subdir stanza =
  Dune_scan.classified_command_sites_with_pins_preserving_multiplicity stanza
  |> List.map ~f:(fun (cwd, _pins, site, command) ->
      let rec program_token = function
        | Dune_scan.Program (token, _) -> Some token
        | Dune_scan.Elsewhere (_, nested) | Dune_scan.Unnameable (_, nested) -> program_token nested
        | Dune_scan.Shell _ -> None
      in
      match command with
      | (Dune_scan.Runs path | Dune_scan.Runs_dependency path) when Dune_scan.is_absolute path ->
          Error (absolute_smoke_error dune_path path)
      | Dune_scan.Runs path
        when Option.exists (program_token site) ~f:Dune_scan.is_path_lookup_token ->
          Error (bare_smoke_error dune_path path)
      | (Dune_scan.Runs _ | Dune_scan.Runs_dependency _) when Dune_scan.is_absolute cwd ->
          Error (absolute_chdir_error dune_path cwd)
      | (Dune_scan.Runs _ | Dune_scan.Runs_dependency _) as command ->
          let local =
            Option.value_exn (Dune_scan.command_file_path ~cwd command)
            |> Dune_scan.in_subdir subdir |> Dune_scan.normalize_path
          in
          if
            allow_verified_helper
            && String.equal local verified_helper_local
            &&
            match site with
            | Dune_scan.Program (command, []) -> String.equal command verified_helper_command
            | _ -> false
          then Ok None
          else Ok (Some (Local local))
      | Dune_scan.Runs_public name -> Ok (Some (Public name))
      | Dune_scan.Unrecognized command
      | Dune_scan.Unknown_directory command
      | Dune_scan.Path_rewritten command ->
          Error (opaque_smoke_error dune_path command)
      | Dune_scan.External -> Error (external_smoke_error dune_path))

let target_name = function Local path -> path | Public name -> "%{bin:" ^ name ^ "}"

(* The per-file walks that run after the main census, each named in its diagnostic so a reader can
   tell which question the file could not answer. *)
let path_rewriting_walk = "PATH-rewriting env stanzas"
let directory_target_walk = "directory targets"
let target_producer_walk = "target producers"
let executable_dependency_walk = "executable link dependencies"

let walk_error ~walk dune_path exn =
  Printf.sprintf "cannot scan %s for %s: %s" dune_path walk (Stdlib.Printexc.to_string exn)

(* [f] over every dune file, concatenated. A file whose walk raises is reported, never dropped: an
   empty answer would read as "nothing to find there", shrinking the census with nothing failing. A
   file the main census cannot parse is reported once per walk as well, which is redundant but keeps
   each walk's coverage its own rather than an inference about what the census caught. *)
let walk_each dune_files ~walk ~f =
  let found, errors =
    List.partition_map dune_files ~f:(fun (dune_path, content) ->
        match f dune_path content with
        | found -> First found
        | exception exn -> Second (walk_error ~walk dune_path exn))
  in
  (List.concat found, errors)

let scan dune_files =
  let declarations, _executable_locals, alias_nodes, generated_targets, scan_errors =
    List.fold dune_files ~init:([], [], [], [], [])
      ~f:(fun
          (all_declarations, all_locals, all_nodes, all_targets, all_errors) (dune_path, content) ->
        try
          let stanzas = Dune_scan.stanzas content in
          let directory = path_dirname dune_path in
          let walked =
            Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
                [ (Dune_scan.in_subdir directory subdir, stanza) ])
          in
          let declarations, executable_locals, declaration_errors =
            if String.equal dune_path "bin/dune" || String.is_prefix dune_path ~prefix:"bin/" then
              List.fold walked ~init:([], [], [])
                ~f:(fun (declarations, locals, errors) (subdir, stanza) ->
                  let found_declarations, found_errors = declarations_of_stanza ~subdir stanza in
                  let locals =
                    match Dune_scan.head stanza with
                    | Some ("executable" | "executables") ->
                        List.rev_append
                          (List.map (Dune_scan.names_of stanza) ~f:(fun name ->
                               Dune_scan.normalize_path (Dune_scan.in_subdir subdir (name ^ ".exe"))))
                          locals
                    | _ -> locals
                  in
                  ( List.rev_append found_declarations declarations,
                    locals,
                    List.rev_append found_errors errors ))
            else ([], [], [])
          in
          let alias_nodes =
            List.filter_mapi walked ~f:(fun index (subdir, stanza) ->
                let names = aliases_of stanza in
                if List.is_empty names then None
                else
                  let dependencies, dependency_errors = alias_dependencies ~subdir stanza in
                  Some
                    {
                      id = Printf.sprintf "%s:%d" dune_path index;
                      dune_path;
                      subdir;
                      stanza;
                      aliases = List.map names ~f:(alias_key ~subdir);
                      is_bin_smoke = List.mem names "bin-smoke" ~equal:String.equal;
                      dependencies;
                      target_dependencies = target_dependencies ~subdir stanza;
                      dependency_errors;
                    })
          in
          let generated_targets =
            List.concat_map walked ~f:(fun (subdir, stanza) -> produced_targets ~subdir stanza)
          in
          let include_errors =
            List.concat_map walked ~f:(fun (_subdir, stanza) ->
                match Dune_scan.head stanza with
                | Some "include" -> [ include_error dune_path ]
                | Some "data_only_dirs" -> [ data_only_dirs_error dune_path ]
                | Some "install" when installs_into_bin stanza -> [ bin_install_error dune_path ]
                | Some "library" when has_action_preprocessor stanza ->
                    [ library_action_preprocess_error dune_path ]
                | _ -> [])
          in
          let inference_errors =
            List.concat_map walked ~f:(fun (_subdir, stanza) ->
                (if unresolved_target_chdir stanza then [ unresolved_target_chdir_error dune_path ]
                 else [])
                @
                if unresolved_inferred_target stanza then
                  [ unresolved_inferred_target_error dune_path ]
                else [])
          in
          ( List.rev_append declarations all_declarations,
            List.rev_append executable_locals all_locals,
            List.rev_append alias_nodes all_nodes,
            List.rev_append generated_targets all_targets,
            List.rev_append declaration_errors
              (List.rev_append include_errors (List.rev_append inference_errors all_errors)) )
        with exn ->
          ( all_declarations,
            all_locals,
            all_nodes,
            all_targets,
            Printf.sprintf "cannot parse %s: %s" dune_path (Exn.to_string exn) :: all_errors ))
  in
  let path_rewriting_directories, path_rewriting_walk_errors =
    walk_each dune_files ~walk:path_rewriting_walk ~f:(fun dune_path content ->
        List.map (Dune_scan.path_rewriting_stanza_scopes content) ~f:(fun subdir ->
            (Dune_scan.in_subdir (path_dirname dune_path) subdir, dune_path)))
  in
  let generated_directory_targets, directory_target_walk_errors =
    walk_each dune_files ~walk:directory_target_walk ~f:(fun dune_path content ->
        let directory = path_dirname dune_path in
        Dune_scan.walk "" (Dune_scan.stanzas content) ~f:(fun subdir stanza ->
            produced_directory_targets ~subdir:(Dune_scan.in_subdir directory subdir) stanza))
  in
  let generated_directory_targets =
    List.dedup_and_sort generated_directory_targets ~compare:String.compare
  in
  let target_producer_sites, target_producer_walk_errors =
    walk_each dune_files ~walk:target_producer_walk ~f:(fun dune_path content ->
        let directory = path_dirname dune_path in
        Dune_scan.walk "" (Dune_scan.stanzas content) ~f:(fun subdir stanza ->
            let subdir = Dune_scan.in_subdir directory subdir in
            let targets = produced_targets ~subdir stanza in
            if List.is_empty targets then [] else [ (dune_path, subdir, stanza, targets) ]))
  in
  let target_producers, target_producer_command_errors =
    List.fold target_producer_sites ~init:([], [])
      ~f:(fun (producers, errors) (dune_path, subdir, stanza, targets) ->
        smoke_targets_of_stanza ~allow_verified_helper:false ~dune_path ~subdir stanza
        |> List.fold ~init:(producers, errors) ~f:(fun (producers, errors) -> function
          | Ok (Some target) -> ((dune_path, targets, target) :: producers, errors)
          | Ok None -> (producers, errors)
          | Error error -> (producers, target_producer_command_error dune_path error :: errors)))
  in
  let produces_public_source targets =
    List.exists targets ~f:(fun target ->
        String.is_prefix target ~prefix:"bin/"
        && (String.is_suffix target ~suffix:".ml" || String.is_suffix target ~suffix:".mli"))
  in
  let smoke_roots = List.filter alias_nodes ~f:(fun node -> node.is_bin_smoke) in
  let directory_path_errors node =
    List.filter_map path_rewriting_directories ~f:(fun (directory, dune_path) ->
        if
          String.is_empty directory
          || String.equal directory node.subdir
          || String.is_prefix node.subdir ~prefix:(directory ^ "/")
        then Some (directory_path_error dune_path)
        else None)
  in
  let resolve_alias_dependencies dune_path dependencies =
    List.fold dependencies ~init:([], []) ~f:(fun (nodes, errors) dependency ->
        let errors =
          if may_have_implicit_contributors dependency then
            implicit_alias_error dune_path dependency :: errors
          else errors
        in
        let found =
          List.filter alias_nodes ~f:(fun candidate ->
              List.mem candidate.aliases dependency ~equal:String.equal)
        in
        if List.is_empty found then
          ( nodes,
            Printf.sprintf "%s: @bin-smoke depends on undefined alias %s" dune_path dependency
            :: errors )
        else (List.rev_append found nodes, errors))
  in
  let target_is_public = function
    | Local path ->
        List.exists declarations ~f:(fun declaration -> String.equal declaration.local path)
    | Public name ->
        List.exists declarations ~f:(fun declaration -> String.equal declaration.public name)
  in
  let rec producer_alias_effects visited node =
    if Set.mem visited node.id then ([], [])
    else
      let visited = Set.add visited node.id in
      let direct_targets, direct_errors =
        smoke_targets_of_stanza ~allow_verified_helper:false ~dune_path:node.dune_path
          ~subdir:node.subdir node.stanza
        |> List.fold ~init:([], []) ~f:(fun (targets, errors) -> function
          | Ok (Some target) when target_is_public target -> (target :: targets, errors)
          | Ok (Some target) -> (targets, private_producer_run_error node.dune_path target :: errors)
          | Ok None -> (targets, errors)
          | Error error -> (targets, error :: errors))
      in
      let direct_errors =
        List.rev_append
          (List.map node.dependency_errors
             ~f:(target_producer_dependency_spec_error node.dune_path))
          direct_errors
      in
      let dependencies, resolution_errors =
        resolve_alias_dependencies node.dune_path node.dependencies
      in
      let direct_errors =
        List.rev_append
          (List.map resolution_errors ~f:(target_producer_dependency_spec_error node.dune_path))
          direct_errors
      in
      List.fold dependencies ~init:(direct_targets, direct_errors)
        ~f:(fun (targets, errors) dependency ->
          let found_targets, found_errors = producer_alias_effects visited dependency in
          (List.rev_append found_targets targets, List.rev_append found_errors errors))
  in
  let target_producer_dependency_errors =
    List.concat_map target_producer_sites ~f:(fun (dune_path, subdir, stanza, targets) ->
        if not (produces_public_source targets) then []
        else
          let dependencies, dependency_errors = alias_dependencies ~subdir stanza in
          let found, resolution_errors = resolve_alias_dependencies dune_path dependencies in
          List.map
            (List.rev_append dependency_errors resolution_errors)
            ~f:(target_producer_dependency_spec_error dune_path)
          @ List.concat_map found ~f:(fun node ->
              let targets, errors = producer_alias_effects (Set.empty (module String)) node in
              List.map targets ~f:(target_producer_dependency_error dune_path) @ errors))
  in
  let executable_dependency_errors, executable_dependency_walk_errors =
    walk_each dune_files ~walk:executable_dependency_walk ~f:(fun dune_path content ->
        if not (String.equal dune_path "bin/dune" || String.is_prefix dune_path ~prefix:"bin/") then
          []
        else
          let directory = path_dirname dune_path in
          Dune_scan.walk "" (Dune_scan.stanzas content) ~f:(fun subdir stanza ->
              let subdir = Dune_scan.in_subdir directory subdir in
              let declarations, _errors = declarations_of_stanza ~subdir stanza in
              if List.is_empty declarations then []
              else
                List.concat_map [ "link_deps"; "preprocessor_deps" ] ~f:(fun field ->
                    let dependencies, dependency_errors =
                      alias_dependencies_in_field ~field ~subdir stanza
                    in
                    let target_dependencies = target_dependencies_in_field ~field ~subdir stanza in
                    let found, resolution_errors =
                      resolve_alias_dependencies dune_path dependencies
                    in
                    List.map
                      (List.rev_append dependency_errors resolution_errors)
                      ~f:(executable_dependency_spec_error dune_path)
                    @ List.concat_map found ~f:(fun node ->
                        let targets, errors =
                          producer_alias_effects (Set.empty (module String)) node
                        in
                        List.map targets ~f:(executable_dependency_error dune_path) @ errors)
                    @ List.concat_map target_producer_sites
                        ~f:(fun (producer_path, producer_subdir, producer_stanza, targets) ->
                          if
                            not
                              (List.exists targets ~f:(fun target ->
                                   List.mem target_dependencies target ~equal:String.equal))
                          then []
                          else
                            let direct_errors =
                              smoke_targets_of_stanza ~allow_verified_helper:false
                                ~dune_path:producer_path ~subdir:producer_subdir producer_stanza
                              |> List.map ~f:(function
                                | Ok (Some target) when target_is_public target ->
                                    executable_dependency_error dune_path target
                                | Ok (Some target) ->
                                    private_producer_run_error producer_path target
                                | Ok None -> ""
                                | Error error -> target_producer_command_error producer_path error)
                              |> List.filter ~f:(Fn.non String.is_empty)
                            in
                            let producer_dependencies, producer_dependency_errors =
                              alias_dependencies ~subdir:producer_subdir producer_stanza
                            in
                            let producer_aliases, producer_resolution_errors =
                              resolve_alias_dependencies producer_path producer_dependencies
                            in
                            direct_errors
                            @ List.map
                                (List.rev_append producer_dependency_errors
                                   producer_resolution_errors)
                                ~f:(executable_dependency_spec_error dune_path)
                            @ List.concat_map producer_aliases ~f:(fun node ->
                                let targets, errors =
                                  producer_alias_effects (Set.empty (module String)) node
                                in
                                List.map targets ~f:(executable_dependency_error dune_path) @ errors)))))
  in
  let rec visit visited targets errors = function
    | [] -> (targets, errors)
    | node :: rest when Set.mem visited node.id -> visit visited targets errors rest
    | node :: rest ->
        let visited = Set.add visited node.id in
        let allow_verified_helper =
          List.equal String.equal node.aliases [ verified_helper_alias ]
        in
        let found =
          smoke_targets_of_stanza ~allow_verified_helper ~dune_path:node.dune_path
            ~subdir:node.subdir node.stanza
        in
        (* Spent only where the stanza runs the pinned probe ONCE and nothing else: an opaque
           command or a second program beside it puts the stanza back under every refusal, so the
           exemption cannot be widened by adding to the rule it was written for. Exactly once, not
           at least once, because this scan preserves repeated command sites on purpose -- dropping
           a duplicated probe as a set would hide the very multiplicity the census is careful to
           keep. *)
        let platform_exempt =
          match platform_contribution_of ~dune_path:node.dune_path node.stanza with
          | None -> false
          | Some contribution ->
              let is_probe = function
                | Ok (Some (Local local)) -> String.equal local contribution.contribution_local
                | _ -> false
              in
              List.count found ~f:is_probe = 1
              && List.for_all found ~f:(function Ok None -> true | found -> is_probe found)
        in
        let targets, command_errors =
          if platform_exempt then (targets, [])
          else
            List.fold found ~init:(targets, []) ~f:(fun (targets, errors) -> function
              | Ok (Some target) -> (target :: targets, errors)
              | Ok None -> (targets, errors)
              | Error error -> (targets, error :: errors))
        in
        let condition_errors =
          if has_enabled_if node.stanza && not platform_exempt then
            [ smoke_condition_error node.dune_path ]
          else []
        in
        let exit_errors =
          if accepts_nonstandard_exit node.stanza then [ accepted_exit_error node.dune_path ]
          else []
        in
        let dynamic_run_errors =
          if contains_head "dynamic-run" node.stanza then [ dynamic_run_error node.dune_path ]
          else []
        in
        let cached_command_alias_errors =
          let command_bearing =
            not
              (List.is_empty
                 (Dune_scan.classified_command_sites_with_pins_preserving_multiplicity node.stanza))
          in
          if command_bearing && not (depends_on_universe node.stanza) then
            [ cached_command_alias_error node.dune_path node.aliases ]
          else []
        in
        let implicit_runner_errors =
          match Dune_scan.head node.stanza with
          | Some ("test" | "tests") -> [ implicit_runner_error node.dune_path ]
          | Some "library" when Option.is_some (Dune_scan.field node.stanza "inline_tests") ->
              [ implicit_runner_error node.dune_path ]
          | _ -> []
        in
        let directory_path_errors = directory_path_errors node in
        let generated_dependency_errors =
          List.filter node.target_dependencies ~f:(fun dependency ->
              List.mem generated_targets dependency ~equal:String.equal
              || List.exists generated_directory_targets ~f:(fun directory ->
                  String.is_prefix dependency ~prefix:(directory ^ "/")))
          |> List.map ~f:(fun dependency ->
              Printf.sprintf "%s: @bin-smoke reaches generated target dependency %s" node.dune_path
                dependency)
        in
        let alias_targets = produced_targets ~subdir:node.subdir node.stanza in
        let cached_alias_errors =
          if List.is_empty alias_targets then []
          else [ cached_alias_error node.dune_path alias_targets ]
        in
        let dependencies, missing_dependency_errors =
          resolve_alias_dependencies node.dune_path node.dependencies
        in
        let errors =
          List.rev_append node.dependency_errors
            (List.rev_append command_errors
               (List.rev_append condition_errors
                  (List.rev_append exit_errors
                     (List.rev_append dynamic_run_errors
                        (List.rev_append cached_command_alias_errors
                           (List.rev_append implicit_runner_errors
                              (List.rev_append directory_path_errors
                                 (List.rev_append generated_dependency_errors
                                    (List.rev_append cached_alias_errors
                                       (List.rev_append missing_dependency_errors errors))))))))))
        in
        visit visited targets errors (List.rev_append dependencies rest)
  in
  let targets, smoke_errors =
    visit
      (Set.empty (module String))
      []
      (List.rev_append executable_dependency_errors
         (List.rev_append target_producer_command_errors target_producer_dependency_errors))
      smoke_roots
  in
  let smoke_stanza_count = List.length smoke_roots in
  let scan_errors = List.rev_append smoke_errors scan_errors in
  let declarations = List.sort declarations ~compare:(fun a b -> String.compare a.local b.local) in
  let duplicate_locals =
    duplicates (List.map declarations ~f:(fun declaration -> declaration.local))
  in
  let duplicate_public =
    duplicates (List.map declarations ~f:(fun declaration -> declaration.public))
  in
  let canonicalize target =
    let matches =
      List.filter declarations ~f:(fun declaration ->
          match target with
          | Local path -> String.equal declaration.local path
          | Public name -> String.equal declaration.public name)
    in
    match matches with
    | [ declaration ] -> Ok declaration.local
    | [] -> Error (target_name target)
    | _ -> Error (target_name target ^ " (ambiguous public executable identity)")
  in
  let smoked, unexpected =
    List.fold targets ~init:([], []) ~f:(fun (smoked, unexpected) target ->
        match canonicalize target with
        | Ok local -> (local :: smoked, unexpected)
        | Error target -> (smoked, target :: unexpected))
  in
  let smoked = sorted smoked in
  let expected = List.map declarations ~f:(fun declaration -> declaration.local) |> sorted in
  let missing =
    List.filter expected ~f:(fun local -> not (List.mem smoked local ~equal:String.equal))
  in
  let duplicate_smoked = duplicates smoked in
  let generated_public_run_errors =
    List.filter_map target_producers ~f:(fun (dune_path, targets, target) ->
        match canonicalize target with
        | Ok local -> Some (generated_public_run_error dune_path local targets)
        | Error _ when produces_public_source targets ->
            Some (private_producer_run_error dune_path target)
        | Error _ -> None)
  in
  let errors =
    List.rev scan_errors @ path_rewriting_walk_errors @ directory_target_walk_errors
    @ target_producer_walk_errors @ executable_dependency_walk_errors
    @ (if Int.equal smoke_stanza_count 0 then [ "the repository defines no @bin-smoke action" ]
       else [])
    @ List.map duplicate_locals ~f:(Printf.sprintf "duplicate public local identity: %s")
    @ List.map duplicate_public ~f:(Printf.sprintf "duplicate public installed identity: %s")
    @ List.map duplicate_smoked ~f:(Printf.sprintf "@bin-smoke runs more than once: %s")
    @ generated_public_run_errors
  in
  { declarations; smoked; missing; unexpected = sorted unexpected; errors }

let scan_bin_content content = scan [ ("bin/dune", content) ]

let complete result =
  List.length result.declarations > 0
  && Int.equal (List.length result.missing) 0
  && Int.equal (List.length result.unexpected) 0
  && Int.equal (List.length result.errors) 0
  && List.equal String.equal
       (List.map result.declarations ~f:(fun declaration -> declaration.local) |> sorted)
       result.smoked

let complete_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(executable (name env_spelling_gate))
(rule
 (alias bin-smoke-env_spelling_gate)
 (deps (universe))
 (action (run %{dep:env_spelling_gate.exe})))
(rule
 (alias bin-smoke)
 (deps (alias bin-smoke-env_spelling_gate) (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{bin:beta-tool}))))|dune}

let missing_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let duplicate_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let conditional_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (enabled_if false)
 (action (run %{exe:alpha.exe})))|dune}

let conditional_public_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (enabled_if false))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe})))|dune}

(* The pinned platform contribution and the four ways of not being it. Each is scanned beside a
   complete bin/ census, since the file carrying such a contribution declares no public executable
   of its own. *)
let contribution = List.hd_exn platform_contributions

let scan_contribution ?(dune_path = contribution.contribution_dune) content =
  scan [ ("bin/dune", complete_fixture); (dune_path, content) ]

let platform_contribution_fixture =
  {dune|(executable
 (name metal_queue_probe)
 (enabled_if (= %{system} macosx)))
(rule
 (alias bin-smoke)
 (enabled_if (= %{system} macosx))
 (deps (universe))
 (action (run %{exe:metal_queue_probe.exe} --iterations=1024)))|dune}

let unconditional_contribution_fixture =
  {dune|(executable (name metal_queue_probe))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:metal_queue_probe.exe} --iterations=1024)))|dune}

let foreign_condition_contribution_fixture =
  {dune|(executable
 (name metal_queue_probe)
 (enabled_if (= %{profile} release)))
(rule
 (alias bin-smoke)
 (enabled_if (= %{profile} release))
 (deps (universe))
 (action (run %{exe:metal_queue_probe.exe} --iterations=1024)))|dune}

let duplicated_contribution_fixture =
  {dune|(executable
 (name metal_queue_probe)
 (enabled_if (= %{system} macosx)))
(rule
 (alias bin-smoke)
 (enabled_if (= %{system} macosx))
 (deps (universe))
 (action
  (progn
   (run %{exe:metal_queue_probe.exe} --iterations=1024)
   (run %{exe:metal_queue_probe.exe} --iterations=1024))))|dune}

let widened_contribution_fixture =
  {dune|(executable
 (names metal_queue_probe other_probe)
 (enabled_if (= %{system} macosx)))
(rule
 (alias bin-smoke)
 (enabled_if (= %{system} macosx))
 (deps (universe))
 (action
  (progn
   (run %{exe:metal_queue_probe.exe} --iterations=1024)
   (run %{exe:other_probe.exe}))))|dune}

let included_fixture = complete_fixture ^ "\n(include extra-stanzas.inc)\n"

(* [#;] is a comment to sexplib and an atom to dune, so [Dune_scan.stanzas] refuses this file: every
   per-file walk raises on it, and each has to say so rather than return nothing. *)
let unreadable_fixture =
  complete_fixture ^ "\n(rule #;(alias bin-smoke) (action (run %{exe:alpha.exe})))\n"

let opaque_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (progn
   (run %{exe:alpha.exe})
   (bash "./alpha.exe --again"))))|dune}

let competing_alias_fixture =
  {dune|(rule
 (alias bin-smoke)
 (action (run %{bin:alpha-tool})))|dune}

let transitive_duplicate_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias helper-smoke)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let external_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps runner.py)
 (action
  (progn
   (run %{exe:alpha.exe})
   (run python3 runner.py))))|dune}

let accepted_exit_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (with-accepted-exit-codes 1
   (run %{exe:alpha.exe}))))|dune}

let rerunning_helper_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias helper-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke)))|dune}

let cached_helper_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias helper-smoke)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke)))|dune}

let cached_root_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe})))|dune}

let chdir_pform_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (chdir smoke (run %{exe:alpha.exe}))))|dune}

let chdir_parent_pform_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (chdir smoke (run %{exe:../alpha.exe}))))|dune}

let absolute_chdir_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (chdir /tmp (run ../../bin/alpha.exe))))|dune}

let generated_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action
  (progn
   (run %{exe:alpha.exe})
   (touch %{target}))))
(rule
 (alias bin-smoke)
 (deps smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let inferred_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (with-stdout-to smoke.stamp
   (run %{exe:alpha.exe}))))
(rule
 (alias bin-smoke)
 (deps smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let chdir_inferred_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (chdir generated
   (with-stdout-to smoke.stamp
    (run %{exe:alpha.exe})))))
(rule
 (alias bin-smoke)
 (deps (universe) generated/smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let unresolved_chdir_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (chdir %{context_name}
   (with-stdout-to smoke.stamp
    (run %{exe:alpha.exe})))))
(rule
 (alias bin-smoke)
 (deps (universe) default/smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let pform_inferred_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (progn
   (run %{exe:alpha.exe})
   (touch smoke-%{context_name}.stamp))))
(rule
 (alias bin-smoke)
 (deps (universe) smoke-default.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let target_bearing_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (target smoke.stamp)
 (action
  (progn
   (run %{exe:alpha.exe})
   (touch smoke.stamp))))|dune}

let directory_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (targets (dir smoke-output))
 (action
  (progn
   (run %{exe:alpha.exe})
   (mkdir smoke-output))))
(rule
 (alias bin-smoke)
 (deps smoke-output/stamp)
 (action (run %{exe:alpha.exe})))|dune}

let inferred_directory_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (progn
   (run %{exe:alpha.exe})
   (mkdir smoke-output))))
(rule
 (alias bin-smoke)
 (deps (universe) smoke-output/stamp)
 (action (run %{exe:alpha.exe})))|dune}

let directory_target_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (targets (dir smoke-output))
 (action (run %{exe:alpha.exe})))|dune}

let rewritten_path_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (setenv Path elsewhere
   (run alpha.exe))))|dune}

let absolute_path_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run /../bin/alpha.exe)))|dune}

let bare_path_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run alpha.exe)))|dune}

let dynamic_run_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action (dynamic-run %{exe:alpha.exe})))|dune}

let directory_path_fixture =
  {dune|(env
 (_
  (env-vars
   (Path elsewhere))))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action (run alpha.exe)))|dune}

let directory_binaries_fixture =
  {dune|(env
 (_
  (binaries helper.exe)))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run alpha.exe)))|dune}

let nested_unaffected_path_fixture =
  {dune|(subdir tools
 (env
  (_
   (env-vars
    (Path elsewhere)))))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let nested_affected_path_fixture =
  {dune|(subdir tools
 (env
  (_
   (env-vars
    (Path elsewhere))))
 (rule
  (alias helper-smoke)
  (deps (universe))
  (action (chdir .. (run alpha.exe)))))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (alias tools/helper-smoke)))|dune}

let private_launcher_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name helper))
(rule
 (alias helper-smoke)
 (action (run %{exe:helper.exe} %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let verified_helper_launcher_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name env_spelling_gate))
(rule
 (alias bin-smoke-env_spelling_gate)
 (action (run %{dep:env_spelling_gate.exe} %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias bin-smoke-env_spelling_gate))
 (action (run %{exe:alpha.exe})))|dune}

let action_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe} %{dep:smoke.stamp})))|dune}

let embedded_action_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (echo value=%{read:smoke.stamp}))))|dune}

let included_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe) (include aliases.sexp))
 (action (run %{exe:alpha.exe})))|dune}

let explicit_read_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps %{read:smoke.stamp})
 (action (run %{exe:alpha.exe})))|dune}

let embedded_explicit_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (universe) ./%{path:smoke.stamp})
 (action (run %{exe:alpha.exe})))|dune}

let expanded_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe) %{read-lines:manifest})
 (action (run %{exe:alpha.exe})))|dune}

let unmodeled_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke-default.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (universe) smoke-%{context_name}.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let generated_source_input_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (target alpha.ml)
 (action (run %{exe:beta.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let opaque_generated_source_input_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (target alpha.ml)
 (action (bash "./beta.exe > alpha.ml")))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let generated_source_alias_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (alias source-helper)
 (deps (universe))
 (action (run %{exe:beta.exe})))
(rule
 (target alpha.ml)
 (deps (alias source-helper))
 (action (touch alpha.ml)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let external_generated_source_input_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (target alpha.ml)
 (deps generator.py)
 (action (run python3 generator.py)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let unused_generated_source_alias_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (alias source-helper)
 (deps (universe))
 (action (run %{exe:beta.exe})))
(rule
 (target unused.stamp)
 (deps (alias source-helper))
 (action (touch unused.stamp)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let opaque_generated_source_alias_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (alias source-helper)
 (deps generator.py (universe))
 (action (run python3 generator.py)))
(rule
 (target alpha.ml)
 (deps (alias source-helper))
 (action (touch alpha.ml)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let private_generated_source_input_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(executable (name generator))
(rule
 (target alpha.ml)
 (action (run %{exe:generator.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let private_generated_interface_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(executable (name generator))
(rule
 (target alpha.mli)
 (action (run %{exe:generator.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let included_generated_source_dependency_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (target alpha.ml)
 (deps (include source-deps.sexp))
 (action (touch alpha.ml)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let implicit_generated_source_alias_fixture =
  {dune|(executables
 (names alpha beta)
 (public_names alpha-tool beta-tool))
(rule
 (target alpha.ml)
 (deps (alias runtest-helper))
 (action (touch alpha.ml)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let nested_bin_root_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let nested_bin_declaration_fixture = {dune|(executable (name beta) (public_name beta-tool))|dune}

let library_action_preprocessor_fixture =
  {dune|(library
 (name linked)
 (preprocess
  (action (run %{exe:beta.exe} %{input-file}))))
(executable
 (name alpha)
 (public_name alpha-tool)
 (libraries linked))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let executable_link_dependency_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (link_deps (alias build-helper)))
(executable (name beta) (public_name beta-tool))
(rule
 (alias build-helper)
 (deps (universe))
 (action (run %{exe:beta.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let executable_preprocessor_dependency_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (preprocessor_deps (alias build-helper)))
(executable (name beta) (public_name beta-tool))
(rule
 (alias build-helper)
 (deps (universe))
 (action (run %{exe:beta.exe})))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let executable_link_target_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (link_deps smoke.stamp))
(executable (name beta) (public_name beta-tool))
(rule
 (alias build-helper)
 (deps (universe))
 (action (run %{exe:beta.exe})))
(rule
 (target smoke.stamp)
 (deps (alias build-helper))
 (action (touch smoke.stamp)))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let action_preprocessor_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (preprocess
  (action (run %{exe:beta.exe} %{input-file}))))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let data_only_root_fixture = {dune|(data_only_dirs fixtures)|dune}
let data_only_bin_fixture = {dune|(executable (name alpha) (public_name alpha-tool))|dune}

let data_only_alias_fixture =
  {dune|(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:../bin/alpha.exe})))|dune}

let bin_install_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name helper))
(install
 (section bin)
 (files helper.exe))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let transitive_path_bin_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (alias ../test/helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let transitive_path_helper_fixture =
  {dune|(env (_ (env-vars (Path elsewhere))))
(rule
 (alias helper-smoke)
 (action (chdir ../bin (run alpha.exe))))|dune}

let implicit_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(test
 (name hidden)
 (action (run %{exe:alpha.exe})))
(alias (name runtest))
(rule
 (alias bin-smoke)
 (deps (alias runtest))
 (action (run %{exe:alpha.exe})))|dune}

let builtin_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(alias (name all))
(rule
 (alias bin-smoke)
 (deps (alias all))
 (action (run %{exe:alpha.exe})))|dune}

let literal_action_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (action
  (progn
   (run %{exe:alpha.exe})
   (diff expected smoke.stamp))))|dune}

let chdir_literal_action_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(subdir generated
 (rule
  (target smoke.stamp)
  (action (run %{exe:../alpha.exe}))))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action
  (chdir generated
   (progn
    (run %{exe:alpha.exe})
    (diff expected smoke.stamp)))))|dune}

let conditional_private_fixture =
  {dune|(executables
 (names helper)
 (public_names -)
 (enabled_if false))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (universe))
 (action (run %{exe:alpha.exe})))|dune}

let implicit_runner_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(test
 (name hidden)
 (alias helper-smoke))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let controls_hold () =
  let accepted = scan_bin_content complete_fixture in
  let refused = scan_bin_content missing_fixture in
  let duplicated = scan_bin_content duplicate_fixture in
  let conditional_smoke = scan_bin_content conditional_smoke_fixture in
  let conditional_public = scan_bin_content conditional_public_fixture in
  let included = scan_bin_content included_fixture in
  let unreadable = scan_bin_content unreadable_fixture in
  let walk_reported walk =
    match Dune_scan.stanzas unreadable_fixture with
    | _ -> false
    | exception exn ->
        List.mem unreadable.errors (walk_error ~walk "bin/dune" exn) ~equal:String.equal
  in
  let opaque = scan_bin_content opaque_smoke_fixture in
  let competing =
    scan [ ("bin/dune", complete_fixture); ("test/synthetic/dune", competing_alias_fixture) ]
  in
  let transitive = scan_bin_content transitive_duplicate_fixture in
  let rerunning_helper = scan_bin_content rerunning_helper_fixture in
  let cached_helper = scan_bin_content cached_helper_fixture in
  let cached_root = scan_bin_content cached_root_fixture in
  let chdir_pform = scan_bin_content chdir_pform_fixture in
  let chdir_parent_pform = scan_bin_content chdir_parent_pform_fixture in
  let absolute_chdir = scan_bin_content absolute_chdir_fixture in
  let external_result = scan_bin_content external_smoke_fixture in
  let accepted_exit = scan_bin_content accepted_exit_fixture in
  let generated_target = scan_bin_content generated_target_fixture in
  let inferred_target = scan_bin_content inferred_target_fixture in
  let chdir_inferred_target = scan_bin_content chdir_inferred_target_fixture in
  let unresolved_chdir_target = scan_bin_content unresolved_chdir_target_fixture in
  let pform_inferred_target = scan_bin_content pform_inferred_target_fixture in
  let target_bearing_alias = scan_bin_content target_bearing_alias_fixture in
  let directory_target = scan_bin_content directory_target_fixture in
  let inferred_directory_target = scan_bin_content inferred_directory_target_fixture in
  let directory_target_alias = scan_bin_content directory_target_alias_fixture in
  let rewritten_path = scan_bin_content rewritten_path_fixture in
  let absolute_path = scan_bin_content absolute_path_fixture in
  let bare_path = scan_bin_content bare_path_fixture in
  let dynamic_run = scan_bin_content dynamic_run_fixture in
  let directory_path = scan_bin_content directory_path_fixture in
  let directory_binaries = scan_bin_content directory_binaries_fixture in
  let nested_unaffected_path = scan_bin_content nested_unaffected_path_fixture in
  let nested_affected_path = scan_bin_content nested_affected_path_fixture in
  let private_launcher = scan_bin_content private_launcher_fixture in
  let verified_helper_launcher = scan_bin_content verified_helper_launcher_fixture in
  let action_dependency = scan_bin_content action_dependency_fixture in
  let embedded_action_dependency = scan_bin_content embedded_action_dependency_fixture in
  let included_dependency = scan_bin_content included_dependency_fixture in
  let explicit_read_dependency = scan_bin_content explicit_read_dependency_fixture in
  let embedded_explicit_dependency = scan_bin_content embedded_explicit_dependency_fixture in
  let expanded_dependency = scan_bin_content expanded_dependency_fixture in
  let unmodeled_dependency = scan_bin_content unmodeled_dependency_fixture in
  let generated_source_input = scan_bin_content generated_source_input_fixture in
  let opaque_generated_source_input = scan_bin_content opaque_generated_source_input_fixture in
  let generated_source_alias = scan_bin_content generated_source_alias_fixture in
  let external_generated_source_input = scan_bin_content external_generated_source_input_fixture in
  let unused_generated_source_alias = scan_bin_content unused_generated_source_alias_fixture in
  let opaque_generated_source_alias = scan_bin_content opaque_generated_source_alias_fixture in
  let private_generated_source_input = scan_bin_content private_generated_source_input_fixture in
  let private_generated_interface = scan_bin_content private_generated_interface_fixture in
  let included_generated_source_dependency =
    scan_bin_content included_generated_source_dependency_fixture
  in
  let implicit_generated_source_alias = scan_bin_content implicit_generated_source_alias_fixture in
  let nested_bin_declaration =
    scan
      [ ("bin/dune", nested_bin_root_fixture); ("bin/tools/dune", nested_bin_declaration_fixture) ]
  in
  let library_action_preprocessor = scan_bin_content library_action_preprocessor_fixture in
  let executable_link_dependency = scan_bin_content executable_link_dependency_fixture in
  let executable_preprocessor_dependency =
    scan_bin_content executable_preprocessor_dependency_fixture
  in
  let executable_link_target = scan_bin_content executable_link_target_fixture in
  let action_preprocessor = scan_bin_content action_preprocessor_fixture in
  let data_only =
    scan
      [
        ("dune", data_only_root_fixture);
        ("bin/dune", data_only_bin_fixture);
        ("fixtures/dune", data_only_alias_fixture);
      ]
  in
  let bin_install = scan_bin_content bin_install_fixture in
  let transitive_path =
    scan
      [ ("bin/dune", transitive_path_bin_fixture); ("test/dune", transitive_path_helper_fixture) ]
  in
  let implicit_alias = scan_bin_content implicit_alias_fixture in
  let builtin_alias = scan_bin_content builtin_alias_fixture in
  let literal_action_dependency = scan_bin_content literal_action_dependency_fixture in
  let chdir_literal_action_dependency = scan_bin_content chdir_literal_action_dependency_fixture in
  let conditional_private = scan_bin_content conditional_private_fixture in
  let implicit_runner = scan_bin_content implicit_runner_fixture in
  let platform_contribution = scan_contribution platform_contribution_fixture in
  let unconditional_contribution = scan_contribution unconditional_contribution_fixture in
  let foreign_condition = scan_contribution foreign_condition_contribution_fixture in
  let duplicated_contribution = scan_contribution duplicated_contribution_fixture in
  let widened_contribution = scan_contribution widened_contribution_fixture in
  let foreign_path = scan_contribution ~dune_path:"benchmarks/dune" platform_contribution_fixture in
  complete accepted
  && (not (complete refused))
  && List.equal String.equal refused.missing [ "bin/beta.exe" ]
  && List.is_empty refused.unexpected && List.is_empty refused.errors
  && (not (complete duplicated))
  && List.mem duplicated.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && (not (complete conditional_smoke))
  && List.mem conditional_smoke.errors (smoke_condition_error "bin/dune") ~equal:String.equal
  && (not (complete conditional_public))
  && List.mem conditional_public.errors (public_condition_error [ "alpha" ]) ~equal:String.equal
  && (not (complete included))
  && List.mem included.errors (include_error "bin/dune") ~equal:String.equal
  && (not (complete unreadable))
  && walk_reported path_rewriting_walk
  && walk_reported directory_target_walk
  && walk_reported target_producer_walk
  && walk_reported executable_dependency_walk
  && (not (complete opaque))
  && List.mem opaque.errors
       (opaque_smoke_error "bin/dune" "shell: ./alpha.exe --again")
       ~equal:String.equal
  && (not (complete competing))
  && List.mem competing.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && (not (complete transitive))
  && List.mem transitive.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && complete rerunning_helper
  && (not (complete cached_helper))
  && List.mem cached_helper.errors
       (cached_command_alias_error "bin/dune" [ "bin/helper-smoke" ])
       ~equal:String.equal
  && (not (complete cached_root))
  && List.mem cached_root.errors
       (cached_command_alias_error "bin/dune" [ "bin/bin-smoke" ])
       ~equal:String.equal
  && complete chdir_pform
  && (not (complete chdir_parent_pform))
  && List.mem chdir_parent_pform.unexpected "alpha.exe" ~equal:String.equal
  && (not (complete absolute_chdir))
  && List.mem absolute_chdir.errors (absolute_chdir_error "bin/dune" "/tmp") ~equal:String.equal
  && (not (complete external_result))
  && List.mem external_result.errors (external_smoke_error "bin/dune") ~equal:String.equal
  && (not (complete accepted_exit))
  && List.mem accepted_exit.errors (accepted_exit_error "bin/dune") ~equal:String.equal
  && (not (complete generated_target))
  && List.mem generated_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete inferred_target))
  && List.mem inferred_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete chdir_inferred_target))
  && List.mem chdir_inferred_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/generated/smoke.stamp"
       ~equal:String.equal
  && (not (complete unresolved_chdir_target))
  && List.mem unresolved_chdir_target.errors
       (unresolved_target_chdir_error "bin/dune")
       ~equal:String.equal
  && (not (complete pform_inferred_target))
  && List.mem pform_inferred_target.errors
       (unresolved_inferred_target_error "bin/dune")
       ~equal:String.equal
  && (not (complete target_bearing_alias))
  && List.mem target_bearing_alias.errors
       (cached_alias_error "bin/dune" [ "bin/smoke.stamp" ])
       ~equal:String.equal
  && (not (complete directory_target))
  && List.mem directory_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke-output/stamp"
       ~equal:String.equal
  && (not (complete inferred_directory_target))
  && List.mem inferred_directory_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke-output/stamp"
       ~equal:String.equal
  && (not (complete directory_target_alias))
  && List.mem directory_target_alias.errors
       (cached_alias_error "bin/dune" [ "bin/smoke-output" ])
       ~equal:String.equal
  && (not (complete rewritten_path))
  && List.mem rewritten_path.errors
       (opaque_smoke_error "bin/dune" "alpha.exe, under `(setenv PATH elsewhere ...)`")
       ~equal:String.equal
  && (not (complete absolute_path))
  && List.mem absolute_path.errors
       (absolute_smoke_error "bin/dune" "/../bin/alpha.exe")
       ~equal:String.equal
  && (not (complete bare_path))
  && List.mem bare_path.errors (bare_smoke_error "bin/dune" "alpha.exe") ~equal:String.equal
  && (not (complete dynamic_run))
  && List.mem dynamic_run.errors (dynamic_run_error "bin/dune") ~equal:String.equal
  && (not (complete directory_path))
  && List.mem directory_path.errors (directory_path_error "bin/dune") ~equal:String.equal
  && (not (complete directory_binaries))
  && List.mem directory_binaries.errors (directory_path_error "bin/dune") ~equal:String.equal
  && complete nested_unaffected_path
  && (not (complete nested_affected_path))
  && List.mem nested_affected_path.errors (directory_path_error "bin/dune") ~equal:String.equal
  && (not (complete private_launcher))
  && List.mem private_launcher.unexpected "bin/helper.exe" ~equal:String.equal
  && (not (complete verified_helper_launcher))
  && List.mem verified_helper_launcher.unexpected "bin/env_spelling_gate.exe" ~equal:String.equal
  && (not (complete action_dependency))
  && List.mem action_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete embedded_action_dependency))
  && List.mem embedded_action_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete included_dependency))
  && List.mem included_dependency.errors
       "@bin-smoke reaches an included dependency specification that cannot be expanded"
       ~equal:String.equal
  && (not (complete explicit_read_dependency))
  && List.mem explicit_read_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete embedded_explicit_dependency))
  && List.mem embedded_explicit_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete expanded_dependency))
  && List.mem expanded_dependency.errors
       "@bin-smoke reaches a dependency specification expanded from file contents"
       ~equal:String.equal
  && (not (complete unmodeled_dependency))
  && List.mem unmodeled_dependency.errors
       "@bin-smoke reaches a dependency path computed by an unmodeled pform" ~equal:String.equal
  && (not (complete generated_source_input))
  && List.mem generated_source_input.errors
       (generated_public_run_error "bin/dune" "bin/beta.exe" [ "bin/alpha.ml" ])
       ~equal:String.equal
  && (not (complete opaque_generated_source_input))
  && List.mem opaque_generated_source_input.errors
       (target_producer_command_error "bin/dune"
          (opaque_smoke_error "bin/dune" "shell: ./beta.exe > alpha.ml"))
       ~equal:String.equal
  && (not (complete generated_source_alias))
  && List.mem generated_source_alias.errors
       (target_producer_dependency_error "bin/dune" (Local "bin/beta.exe"))
       ~equal:String.equal
  && (not (complete external_generated_source_input))
  && List.mem external_generated_source_input.errors
       (target_producer_command_error "bin/dune" (external_smoke_error "bin/dune"))
       ~equal:String.equal
  && (not (complete unused_generated_source_alias))
  && List.equal String.equal unused_generated_source_alias.missing [ "bin/beta.exe" ]
  && List.is_empty unused_generated_source_alias.errors
  && (not (complete opaque_generated_source_alias))
  && List.mem opaque_generated_source_alias.errors (external_smoke_error "bin/dune")
       ~equal:String.equal
  && (not (complete private_generated_source_input))
  && List.mem private_generated_source_input.errors
       (private_producer_run_error "bin/dune" (Local "bin/generator.exe"))
       ~equal:String.equal
  && (not (complete private_generated_interface))
  && List.mem private_generated_interface.errors
       (private_producer_run_error "bin/dune" (Local "bin/generator.exe"))
       ~equal:String.equal
  && (not (complete included_generated_source_dependency))
  && List.mem included_generated_source_dependency.errors
       (target_producer_dependency_spec_error "bin/dune"
          "@bin-smoke reaches an included dependency specification that cannot be expanded")
       ~equal:String.equal
  && (not (complete implicit_generated_source_alias))
  && List.mem implicit_generated_source_alias.errors
       (target_producer_dependency_spec_error "bin/dune"
          (implicit_alias_error "bin/dune" "bin/runtest-helper"))
       ~equal:String.equal
  && (not (complete nested_bin_declaration))
  && List.equal String.equal nested_bin_declaration.missing [ "bin/tools/beta.exe" ]
  && (not (complete library_action_preprocessor))
  && List.mem library_action_preprocessor.errors
       (library_action_preprocess_error "bin/dune")
       ~equal:String.equal
  && (not (complete executable_link_dependency))
  && List.mem executable_link_dependency.errors
       (executable_dependency_error "bin/dune" (Local "bin/beta.exe"))
       ~equal:String.equal
  && (not (complete executable_preprocessor_dependency))
  && List.mem executable_preprocessor_dependency.errors
       (executable_dependency_error "bin/dune" (Local "bin/beta.exe"))
       ~equal:String.equal
  && (not (complete executable_link_target))
  && List.mem executable_link_target.errors
       (executable_dependency_error "bin/dune" (Local "bin/beta.exe"))
       ~equal:String.equal
  && (not (complete action_preprocessor))
  && List.mem action_preprocessor.errors
       (public_action_preprocess_error [ "alpha" ])
       ~equal:String.equal
  && (not (complete data_only))
  && List.mem data_only.errors (data_only_dirs_error "dune") ~equal:String.equal
  && (not (complete bin_install))
  && List.mem bin_install.errors (bin_install_error "bin/dune") ~equal:String.equal
  && (not (complete transitive_path))
  && List.mem transitive_path.errors (directory_path_error "test/dune") ~equal:String.equal
  && (not (complete implicit_alias))
  && List.mem implicit_alias.errors
       (implicit_alias_error "bin/dune" "bin/runtest")
       ~equal:String.equal
  && (not (complete builtin_alias))
  && List.mem builtin_alias.errors (implicit_alias_error "bin/dune" "bin/all") ~equal:String.equal
  && (not (complete literal_action_dependency))
  && List.mem literal_action_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete chdir_literal_action_dependency))
  && List.mem chdir_literal_action_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/generated/smoke.stamp"
       ~equal:String.equal
  && complete conditional_private
  && (not (complete implicit_runner))
  && List.mem implicit_runner.errors (implicit_runner_error "bin/dune") ~equal:String.equal
  (* The pinned platform contribution is accepted, and each way of not being it is refused: an
     unconditional run of the same probe is an ordinary private executable, a condition that is not
     the pinned one is a different claim about when the stanza runs, a second program beside the
     probe takes the exemption away, the same stanza in another dune file is another file, and the
     probe run TWICE is the multiplicity this scan exists to keep -- reported, not absorbed. *)
  && complete platform_contribution
  && (not (complete unconditional_contribution))
  && List.mem unconditional_contribution.unexpected contribution.contribution_local
       ~equal:String.equal
  && (not (complete foreign_condition))
  && List.mem foreign_condition.errors
       (smoke_condition_error contribution.contribution_dune)
       ~equal:String.equal
  && List.mem foreign_condition.unexpected contribution.contribution_local ~equal:String.equal
  && (not (complete duplicated_contribution))
  && List.count duplicated_contribution.unexpected ~f:(String.equal contribution.contribution_local)
     = 2
  && (not (complete widened_contribution))
  && List.mem widened_contribution.errors
       (smoke_condition_error contribution.contribution_dune)
       ~equal:String.equal
  && List.mem widened_contribution.unexpected contribution.contribution_local ~equal:String.equal
  && (not (complete foreign_path))
  && List.mem foreign_path.errors (smoke_condition_error "benchmarks/dune") ~equal:String.equal
  && List.mem foreign_path.unexpected "benchmarks/metal_queue_probe.exe" ~equal:String.equal

let report_diagnostics result =
  eprintf "Public bin executables (%d): [%s]\n" (List.length result.declarations)
    (List.map result.declarations ~f:(fun declaration ->
         Printf.sprintf "%s -> %s" declaration.local declaration.public)
    |> String.concat ~sep:"; ");
  eprintf "@bin-smoke members (%d): [%s]\n" (List.length result.smoked)
    (String.concat ~sep:"; " result.smoked);
  List.iter result.missing ~f:(fun local -> eprintf "missing from @bin-smoke: %s\n" local);
  List.iter result.unexpected ~f:(fun target ->
      eprintf "not a public bin executable but run by @bin-smoke: %s\n" target);
  List.iter result.errors ~f:(eprintf "%s\n")

let report result =
  report_diagnostics result;
  p "bin-smoke covers every public bin executable exactly once" (complete result)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--refusal-control" ] ->
      let child = Fresh_process.run [ "--negative-control" ] in
      let ok =
        Fresh_process.matches ~exit:1
          ~contains:[ "bin-smoke covers every public bin executable exactly once: false" ]
          child
      in
      if not ok then Fresh_process.report ~label:"omitted executable refusal" child;
      p "synthetic public executable omitted from @bin-smoke is refused" ok;
      Test_utils.Refusal_control_manifest.print "bin_smoke_membership_scan.ml"
  | [ _; "--negative-control" ] -> report (scan_bin_content missing_fixture)
  | _ :: workspace_root :: dune_paths ->
      let base = Dune_scan.base_dir workspace_root in
      let dune_files =
        List.filter_map dune_paths ~f:(fun path ->
            let relative = Dune_scan.repo_relative base path in
            if String.equal (List.last_exn (String.split relative ~on:'/')) "dune" then
              Some (relative, In_channel.read_all path)
            else None)
      in
      let result = scan dune_files in
      let controls_hold = controls_hold () in
      let result =
        if controls_hold then result
        else { result with errors = "synthetic membership controls failed" :: result.errors }
      in
      report result;
      Test_utils.Refusal_control_manifest.print "bin_smoke_membership_scan.ml"
  | argv ->
      eprintf "Usage: %s <workspace-root> <dune-file...>|--negative-control\n" (List.hd_exn argv);
      Stdlib.exit 2
