(* A per-backend golden is useful only when its family covers every backend OCANNL has.

   Active families come from dune rules whose [.expected] atom reads [ocannl_backend.txt]. Existing
   family files provide the converse census, so deleting every member does not erase an active
   family and leaving an orphan family behind does not hide it. The backend vocabulary comes from
   [Backends.all_of_backend] and [Backends.backend_name], so adding a backend makes an incomplete
   family fail here before dune reports a raw missing-rule error on that backend.

   Every unmarked member must contain exactly one [OCANNL backend: <backend>] line naming the
   backend encoded by its filename. That makes a sibling copied into its place fail even when the
   two backends' remaining output is legitimately byte-identical, and refuses a stale source line
   retained beside an appended destination line.

   A member copied from a golden recorded on another backend remains an explicit exception until the
   daily backend sweep can replace it: its sole certification must name the recorded-on backend, and
   this rigid comment must sit INSIDE the dune rule that references the family:

   ; ocannl-golden-recorded-on: <member>.expected <- <backend> -- <reason>

   The scan validates the target, source backend, uniqueness, reason and containing rule, then
   prints every such member into its own golden. Remove the marker when the member is recorded on
   its own backend. All repository-relative paths remain slash-normalized for Windows goldens. *)

open Base
open Stdio
module Backends = Context.Backends
module Dune_scan = Test_utils.Dune_stanza_scan

type member = { path : string; family : string; backend : string }

type provenance = {
  member : string;
  recorded_on : string;
  reason : string;
  containing_families : string list;
}

type result = {
  families : string list;
  incomplete : (string * string list) list;
  provenance : provenance list;
  errors : string list;
}

let marker = "ocannl-golden-recorded-on:"
let backend_placeholder = "<backend>"
let certification_prefix = "OCANNL backend:"
let certification_line backend = "OCANNL backend: " ^ backend
let sorted = List.sort ~compare:String.compare
let dedup_sorted strings = List.dedup_and_sort strings ~compare:String.compare
let path_components path = String.split_on_chars path ~on:[ '/'; '\\' ]

let path_basename path =
  List.filter (path_components path) ~f:(Fn.non String.is_empty)
  |> List.last |> Option.value ~default:""

let path_dirname path =
  match List.filter (path_components path) ~f:(Fn.non String.is_empty) |> List.rev with
  | [] | [ _ ] -> ""
  | _basename :: reversed_dir -> String.concat ~sep:"/" (List.rev reversed_dir)

let join_path dir path =
  let joined = if String.is_empty dir then path else dir ^ "/" ^ path in
  Dune_scan.repo_relative [] joined

let find_occurrences text ~pattern =
  let pattern_length = String.length pattern in
  let rec loop from acc =
    match String.substr_index text ~pos:from ~pattern with
    | None -> List.rev acc
    | Some index -> loop (index + pattern_length) (index :: acc)
  in
  if Int.equal pattern_length 0 then [] else loop 0 []

let replace_backend_placeholder family backend =
  String.substr_replace_first family ~pattern:backend_placeholder ~with_:backend

let member_of_path ~backends path =
  let basename = path_basename path in
  let dirname = path_dirname path in
  let candidates =
    List.concat_map backends ~f:(fun backend ->
        let token = "-" ^ backend in
        find_occurrences basename ~pattern:token
        |> List.filter_map ~f:(fun index ->
            let suffix_at = index + String.length token in
            if
              suffix_at < String.length basename
              && (Char.equal basename.[suffix_at] '-' || Char.equal basename.[suffix_at] '.')
            then
              let family_basename =
                String.prefix basename index ^ "-" ^ backend_placeholder
                ^ String.drop_prefix basename suffix_at
              in
              Some { path; family = join_path dirname family_basename; backend }
            else None))
  in
  match candidates with
  | [] -> Ok None
  | [ member ] -> Ok (Some member)
  | _ ->
      Error
        (Printf.sprintf "%s names more than one backend position: [%s]" path
           (List.map candidates ~f:(fun member -> member.backend)
           |> sorted |> String.concat ~sep:"; "))

let is_backend_read_pform pform =
  match String.chop_prefix pform ~prefix:"read:" with
  | Some path -> String.equal (path_basename path) "ocannl_backend.txt"
  | None -> false

let family_of_rule_atom ~directory atom =
  if not (String.is_suffix atom ~suffix:".expected") then Ok None
  else
    let backend_reads =
      List.count (Dune_scan.pieces atom) ~f:(function
        | Dune_scan.Pform pform -> is_backend_read_pform pform
        | Dune_scan.Literal _ -> false)
    in
    if Int.equal backend_reads 0 then Ok None
    else if backend_reads > 1 then
      Error (Printf.sprintf "expected atom `%s` reads ocannl_backend.txt more than once" atom)
    else
      let rendered, foreign_pform =
        List.fold (Dune_scan.pieces atom) ~init:([], false)
          ~f:(fun (parts, foreign_pform) -> function
          | Dune_scan.Literal text -> (text :: parts, foreign_pform)
          | Dune_scan.Pform pform when is_backend_read_pform pform ->
              (backend_placeholder :: parts, foreign_pform)
          | Dune_scan.Pform _ -> (parts, true))
      in
      if foreign_pform then
        Error
          (Printf.sprintf
             "expected atom `%s` mixes the backend read with another pform, so its family path is \
              unresolved"
             atom)
      else Ok (Some (join_path directory (String.concat (List.rev rendered))))

let families_of_stanza ~dune_path (stanza : Dune_scan.marked_stanza) =
  if not (String.equal stanza.marked_head "rule") then ([], [])
  else
    let directory = join_path (path_dirname dune_path) stanza.marked_subdir in
    List.fold (Dune_scan.atoms stanza.marked_sexp) ~init:([], []) ~f:(fun (families, errors) atom ->
        match family_of_rule_atom ~directory atom with
        | Ok None -> (families, errors)
        | Ok (Some family) -> (family :: families, errors)
        | Error error ->
            (families, Printf.sprintf "%s:%d: %s" dune_path stanza.marked_line error :: errors))
    |> fun (families, errors) -> (dedup_sorted families, List.rev errors)

let split_once text ~on =
  match String.substr_index text ~pattern:on with
  | None -> None
  | Some index -> Some (String.prefix text index, String.drop_prefix text (index + String.length on))

let parse_provenance_declaration ~backends ~directory ~containing_families ~declaration ~reason =
  match split_once declaration ~on:" <- " with
  | None -> Error [ "provenance marker has no ` <member>.expected <- <backend>`" ]
  | Some (basename, recorded_on) ->
      let basename = String.strip basename in
      let recorded_on = String.strip recorded_on in
      let member = join_path directory basename in
      let errors =
        List.filter_opt
          [
            (if
               String.is_empty basename || String.contains basename '/'
               || String.contains basename '\\'
               || not (String.is_suffix basename ~suffix:".expected")
             then Some "provenance target must be an .expected basename in the rule's directory"
             else None);
            (if List.mem backends recorded_on ~equal:String.equal then None
             else Some (Printf.sprintf "recorded-on backend `%s` is not one OCANNL has" recorded_on));
          ]
      in
      if List.is_empty errors then Ok { member; recorded_on; reason; containing_families }
      else Error errors

let marker_issue_errors ~dune_path = function
  | Dune_scan.Malformed_marker { issue_line; issue_malformed; _ } ->
      let detail =
        match issue_malformed with
        | Dune_scan.Missing_reason_separator -> "provenance marker has no ` -- <reason>`"
        | Dune_scan.Short_reason _ -> "provenance marker reason must say why in more than one word"
        | Dune_scan.Repeated_sentinel ->
            Printf.sprintf "more than one `%s` marker occurs on this line" marker
        | Dune_scan.Declaration_error why -> why
      in
      [ Printf.sprintf "%s:%d: %s" dune_path issue_line detail ]
  | Dune_scan.Marker_in_wrong_stanza { issue_why; _ } -> [ issue_why ]
  | Dune_scan.Marker_outside_stanza _ | Dune_scan.Marker_outside_comment _ -> []

let scan ~backends ~expected_files ~dune_files =
  let expected_paths = List.map expected_files ~f:fst in
  let members, member_errors =
    List.fold expected_paths ~init:([], []) ~f:(fun (members, errors) path ->
        match member_of_path ~backends path with
        | Ok None -> (members, errors)
        | Ok (Some member) -> (member :: members, errors)
        | Error error -> (members, error :: errors))
  in
  let rule_families, provenance, dune_errors =
    List.fold dune_files ~init:([], [], [])
      ~f:(fun (all_families, all_provenance, all_errors) (dune_path, content) ->
        match
          Dune_scan.contained_marker_contract content ~sentinel:marker
            ~parse_declaration:(fun stanza ~declaration ~reason ->
              let containing_families, _errors = families_of_stanza ~dune_path stanza in
              let directory = join_path (path_dirname dune_path) stanza.marked_subdir in
              parse_provenance_declaration ~backends ~directory ~containing_families ~declaration
                ~reason)
            ~belongs:(fun _stanza provenance ->
              match member_of_path ~backends provenance.member with
              | Ok (Some member)
                when not (List.mem provenance.containing_families member.family ~equal:String.equal)
                ->
                  Error
                    [
                      Printf.sprintf
                        "%s: provenance marker is not inside the dune rule that references family \
                         %s"
                        provenance.member member.family;
                    ]
              | Ok None | Ok (Some _) | Error _ -> Ok ())
        with
        | contract ->
            let families, provenance, errors =
              List.fold contract.contract_stanzas ~init:([], [], [])
                ~f:(fun (families, provenance, errors) marked ->
                  let stanza = marked.marker_stanza in
                  let stanza_families, family_errors = families_of_stanza ~dune_path stanza in
                  let found =
                    List.map marked.stanza_markers ~f:(fun contained -> contained.contained_value)
                  in
                  ( List.rev_append stanza_families families,
                    List.rev_append found provenance,
                    List.rev_append family_errors errors ))
            in
            let contract_errors =
              List.concat_map contract.contract_issues ~f:(marker_issue_errors ~dune_path)
            in
            (* The contract counts the placed occurrences itself: this check only compares the dumb
               text count against it, so a marker written outside a comment and one written between
               stanzas are the same report -- the author declared something no stanza carries. *)
            let inside_stanzas = contract.contract_stanza_occurrences in
            let errors =
              if Int.equal contract.contract_text_occurrences inside_stanzas then
                List.rev_append contract_errors errors
              else
                Printf.sprintf
                  "%s: found %d `%s` occurrence(s) in the file but only %d inside dune stanzas"
                  dune_path contract.contract_text_occurrences marker inside_stanzas
                :: List.rev_append contract_errors errors
            in
            ( List.rev_append families all_families,
              List.rev_append provenance all_provenance,
              List.rev_append errors all_errors )
        | exception exn ->
            ( all_families,
              all_provenance,
              Printf.sprintf "%s: cannot parse dune structure: %s" dune_path (Exn.to_string exn)
              :: all_errors ))
  in
  let file_families = List.map members ~f:(fun member -> member.family) in
  let families = dedup_sorted (List.rev_append rule_families file_families) in
  let incomplete =
    List.filter_map families ~f:(fun family ->
        let actual =
          List.filter backends ~f:(fun backend ->
              List.mem expected_paths
                (replace_backend_placeholder family backend)
                ~equal:String.equal)
        in
        if List.equal String.equal actual backends then None else Some (family, actual))
  in
  let members_by_path =
    List.fold members
      ~init:(Map.empty (module String))
      ~f:(fun by_path member -> Map.add_exn by_path ~key:member.path ~data:member)
  in
  let marker_targets =
    List.fold provenance
      ~init:(Map.empty (module String))
      ~f:(fun targets marker -> Map.add_multi targets ~key:marker.member ~data:marker)
  in
  let provenance_errors =
    List.concat_map (Map.to_alist marker_targets) ~f:(fun (path, markers) ->
        let marker = List.hd_exn markers in
        let duplicate =
          if List.length markers > 1 then
            [ Printf.sprintf "%s: more than one provenance marker names this member" path ]
          else []
        in
        match Map.find members_by_path path with
        | None ->
            Printf.sprintf "%s: provenance marker target is not a backend golden family member" path
            :: duplicate
        | Some member when String.equal member.backend marker.recorded_on ->
            Printf.sprintf
              "%s: provenance marker records the member on its own backend; remove the marker" path
            :: duplicate
        | Some _ -> duplicate)
  in
  let expected_contents = Map.of_alist_exn (module String) expected_files in
  let certification_errors =
    List.filter_map members ~f:(fun member ->
        let certified_backend =
          match Map.find marker_targets member.path with
          | Some [ provenance ] -> provenance.recorded_on
          | None | Some _ -> member.backend
        in
        let expected_line = certification_line certified_backend in
        let actual_lines =
          Map.find_exn expected_contents member.path |> String.split_lines |> fun lines ->
          List.filter lines ~f:(String.is_prefix ~prefix:certification_prefix)
        in
        if List.equal String.equal actual_lines [ expected_line ] then None
        else
          Some
            (Printf.sprintf
               "%s: backend golden member must contain exactly certification line `%s`; found [%s]"
               member.path expected_line
               (String.concat ~sep:"; " actual_lines)))
  in
  let errors =
    List.rev_append member_errors
      (List.rev_append dune_errors (List.rev_append provenance_errors certification_errors))
    |> List.rev
  in
  let errors =
    if List.is_empty families then "no backend golden family was found in rules or files" :: errors
    else errors
  in
  { families; incomplete; provenance; errors }

let control_results () =
  let backends = [ "cc"; "metal" ] in
  let complete_files =
    [ "test/synthetic/probe-cc.expected"; "test/synthetic/probe-metal.expected" ]
  in
  let native_expected_files =
    List.map complete_files ~f:(fun path ->
        let backend = if String.is_substring path ~substring:"-metal." then "metal" else "cc" in
        (path, certification_line backend ^ "\n"))
  in
  let marked_expected_files =
    List.map complete_files ~f:(fun path -> (path, certification_line "cc" ^ "\n"))
  in
  let family_rule ?marker () =
    Printf.sprintf
      {dune|(rule
 %s
 (deps "probe-%%{read:../config/ocannl_backend.txt}.expected")
 (action (diff "probe-%%{read:../config/ocannl_backend.txt}.expected" probe.actual)))|dune}
      (Option.value marker ~default:"")
  in
  let valid_marker =
    "; ocannl-golden-recorded-on: probe-metal.expected <- cc -- copied from shared golden"
  in
  let complete =
    scan ~backends ~expected_files:marked_expected_files
      ~dune_files:[ ("test/synthetic/dune", family_rule ~marker:valid_marker ()) ]
  in
  let incomplete =
    scan ~backends
      ~expected_files:[ ("test/synthetic/probe-cc.expected", certification_line "cc" ^ "\n") ]
      ~dune_files:[ ("test/synthetic/dune", family_rule ()) ]
  in
  let empty =
    scan ~backends ~expected_files:[] ~dune_files:[ ("test/synthetic/dune", family_rule ()) ]
  in
  let copied_sibling =
    scan ~backends ~expected_files:marked_expected_files
      ~dune_files:[ ("test/synthetic/dune", family_rule ()) ]
  in
  let conflicting_certifications =
    scan ~backends
      ~expected_files:
        [
          ("test/synthetic/probe-cc.expected", certification_line "cc" ^ "\n");
          ( "test/synthetic/probe-metal.expected",
            String.concat ~sep:"\n" [ certification_line "cc"; certification_line "metal"; "" ] );
        ]
      ~dune_files:[ ("test/synthetic/dune", family_rule ()) ]
  in
  let duplicate_certifications =
    scan ~backends
      ~expected_files:
        [
          ("test/synthetic/probe-cc.expected", certification_line "cc" ^ "\n");
          ( "test/synthetic/probe-metal.expected",
            String.concat ~sep:"\n" [ certification_line "metal"; certification_line "metal"; "" ]
          );
        ]
      ~dune_files:[ ("test/synthetic/dune", family_rule ()) ]
  in
  let misplaced =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:[ ("test/synthetic/dune", valid_marker ^ "\n" ^ family_rule ()) ]
  in
  let wrong_rule =
    let content =
      String.concat ~sep:"\n"
        [
          Printf.sprintf
            {dune|(rule
 %s
 (deps "other-%%{read:../config/ocannl_backend.txt}.expected"))|dune}
            valid_marker;
          family_rule ();
        ]
    in
    scan ~backends
      ~expected_files:
        (native_expected_files
        @ [
            ("test/synthetic/other-cc.expected", certification_line "cc" ^ "\n");
            ("test/synthetic/other-metal.expected", certification_line "metal" ^ "\n");
          ])
      ~dune_files:[ ("test/synthetic/dune", content) ]
  in
  let malformed =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule
              ~marker:
                "; ocannl-golden-recorded-on: probe-metal.expected cc -- copied from shared golden"
              () );
        ]
  in
  let missing_separator =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule
              ~marker:
                "; ocannl-golden-recorded-on: probe-metal.expected <- cc copied from shared golden"
              () );
        ]
  in
  let short_reason =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule ~marker:"; ocannl-golden-recorded-on: probe-metal.expected <- cc -- copied"
              () );
        ]
  in
  let duplicated =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule
              ~marker:
                "; ocannl-golden-recorded-on: probe-metal.expected <- cc -- copied from shared \
                 golden ocannl-golden-recorded-on: probe-metal.expected <- cc -- repeated here"
              () );
        ]
  in
  let invalid_target =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule
              ~marker:
                "; ocannl-golden-recorded-on: nested/probe-metal.expected <- cc -- copied from \
                 shared golden"
              () );
        ]
  in
  let invalid_backend =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule
              ~marker:
                "; ocannl-golden-recorded-on: probe-metal.expected <- cuda -- copied from shared \
                 golden"
              () );
        ]
  in
  let outside_comment =
    scan ~backends ~expected_files:native_expected_files
      ~dune_files:
        [
          ( "test/synthetic/dune",
            family_rule ()
            ^ {dune|
(rule (action (echo "ocannl-golden-recorded-on: probe-metal.expected <- cc -- copied from shared golden")))|dune}
          );
        ]
  in
  let expected_family = "test/synthetic/probe-<backend>.expected" in
  let has_exact_error result error = List.mem result.errors error ~equal:String.equal in
  let checks =
    [
      ( "a complete synthesized family and valid contained marker are accepted",
        List.equal String.equal complete.families [ expected_family ]
        && List.is_empty complete.incomplete && List.is_empty complete.errors
        && List.length complete.provenance = 1 );
      ( "a synthesized family missing one backend is refused",
        match incomplete.incomplete with
        | [ (family, actual) ] ->
            String.equal family expected_family && List.equal String.equal actual [ "cc" ]
        | _ -> false );
      ( "a rule whose whole family is absent is refused",
        match empty.incomplete with
        | [ (family, actual) ] -> String.equal family expected_family && List.is_empty actual
        | _ -> false );
      ( "a copied sibling without its member's backend certification is refused",
        has_exact_error copied_sibling
          "test/synthetic/probe-metal.expected: backend golden member must contain exactly \
           certification line `OCANNL backend: metal`; found [OCANNL backend: cc]" );
      ( "conflicting backend certifications are refused",
        has_exact_error conflicting_certifications
          "test/synthetic/probe-metal.expected: backend golden member must contain exactly \
           certification line `OCANNL backend: metal`; found [OCANNL backend: cc; OCANNL backend: \
           metal]" );
      ( "duplicate backend certifications are refused",
        has_exact_error duplicate_certifications
          "test/synthetic/probe-metal.expected: backend golden member must contain exactly \
           certification line `OCANNL backend: metal`; found [OCANNL backend: metal; OCANNL \
           backend: metal]" );
      ( "repository-relative family paths stay slash-normalized",
        String.equal
          (join_path "test\\synthetic" ("probe-" ^ backend_placeholder ^ ".expected"))
          expected_family );
      ( "a marker outside every stanza is refused",
        has_exact_error misplaced
          "test/synthetic/dune: found 1 `ocannl-golden-recorded-on:` occurrence(s) in the file but \
           only 0 inside dune stanzas" );
      ( "a marker inside the wrong family rule is refused",
        has_exact_error wrong_rule
          "test/synthetic/probe-metal.expected: provenance marker is not inside the dune rule that \
           references family test/synthetic/probe-<backend>.expected" );
      ( "a malformed provenance relationship is refused",
        has_exact_error malformed
          "test/synthetic/dune:2: provenance marker has no ` <member>.expected <- <backend>`" );
      ( "a missing reason separator is refused",
        has_exact_error missing_separator
          "test/synthetic/dune:2: provenance marker has no ` -- <reason>`" );
      ( "a one-word provenance reason is refused",
        has_exact_error short_reason
          "test/synthetic/dune:2: provenance marker reason must say why in more than one word" );
      ( "two provenance declarations in one comment are refused",
        has_exact_error duplicated
          "test/synthetic/dune:2: more than one `ocannl-golden-recorded-on:` marker occurs on this \
           line" );
      ( "a provenance target outside the rule directory is refused",
        has_exact_error invalid_target
          "test/synthetic/dune:2: provenance target must be an .expected basename in the rule's \
           directory" );
      ( "an unknown recorded-on backend is refused",
        has_exact_error invalid_backend
          "test/synthetic/dune:2: recorded-on backend `cuda` is not one OCANNL has" );
      ( "a provenance sentinel outside a comment is refused",
        has_exact_error outside_comment
          "test/synthetic/dune: found 1 `ocannl-golden-recorded-on:` occurrence(s) in the file but \
           only 0 inside dune stanzas" );
    ]
  in
  let refusal_diagnostics =
    [
      ("outside every stanza", misplaced.errors);
      ("copied sibling", copied_sibling.errors);
      ("conflicting certifications", conflicting_certifications.errors);
      ("duplicate certifications", duplicate_certifications.errors);
      ("inside the wrong family rule", wrong_rule.errors);
      ("malformed relationship", malformed.errors);
      ("missing reason separator", missing_separator.errors);
      ("one-word reason", short_reason.errors);
      ("duplicated declaration", duplicated.errors);
      ("invalid target", invalid_target.errors);
      ("unknown backend", invalid_backend.errors);
      ("outside a comment", outside_comment.errors);
    ]
  in
  (checks, refusal_diagnostics)

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <expected-or-dune-file...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Dune_scan.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune_scan.repo_relative base path, path))
  in
  let backends = List.map Backends.all_of_backend ~f:Backends.backend_name |> sorted in
  let expected_paths =
    List.filter_map arguments ~f:(fun (relative, path) ->
        if String.is_suffix relative ~suffix:".expected" then
          Some (relative, In_channel.read_all path)
        else None)
  in
  let dune_files =
    List.filter_map arguments ~f:(fun (relative, path) ->
        if String.equal (path_basename relative) "dune" then
          Some (relative, In_channel.read_all path)
        else None)
  in
  let result = scan ~backends ~expected_files:expected_paths ~dune_files in
  List.iter result.incomplete ~f:(fun (family, actual) ->
      eprintf "%s: backend golden family is incomplete\n" family;
      eprintf "  expected backends: [%s]\n" (String.concat ~sep:"; " backends);
      eprintf "  actual backends:   [%s]\n" (String.concat ~sep:"; " actual));
  List.iter result.errors ~f:(eprintf "%s\n");
  let controls, refusal_diagnostics = control_results () in
  List.filter controls ~f:(fun (_, held) -> not held)
  |> List.iter ~f:(fun (name, _) -> eprintf "synthetic control failed: %s\n" name);
  printf "Backend golden families:\n";
  List.iter result.families ~f:(printf "  %s\n");
  printf "\nMembers not yet recorded on their own backend:\n";
  List.sort result.provenance ~compare:(fun a b -> String.compare a.member b.member)
  |> List.iter ~f:(fun { member; recorded_on; reason; _ } ->
      printf "  %s <- %s -- %s\n" member recorded_on reason);
  printf "\nSynthetic controls:\n";
  List.iter controls ~f:(fun (name, _) -> printf "  %s\n" name);
  printf "\nProvenance-marker refusal diagnostics exercised by those controls:\n";
  List.iter refusal_diagnostics ~f:(fun (name, errors) ->
      List.iter errors ~f:(fun error -> printf "  %s -- %s\n" name error));
  printf "\n";
  let complete =
    (not (List.is_empty controls))
    && List.is_empty result.incomplete && List.is_empty result.errors
    && List.for_all controls ~f:snd
  in
  Verdict.p
    "backend golden families are complete, self-certifying, and provenance markers are valid"
    complete;
  Test_utils.Refusal_control_manifest.print "backend_golden_family_scan.ml"
