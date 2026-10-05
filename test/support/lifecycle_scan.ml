open Base
(** Resolution-free lifecycle membership (gh-ocannl-798).

    The instrumentation shares a library with ordinary IR users, so its library alone cannot
    identify probes. A local [ocannl-lifecycle: name[,name...] -- reason] comment declares intent on
    named tests, executables or inline-test libraries. Names must belong to that stanza and its
    libraries must reach the actual instrumentation owner in [arrayjit/lib/dune]. We inspect no
    OCaml references. Unsupported library forms on a declared probe or its dependency path fail
    explicitly, as do misplaced, duplicate and malformed markers. *)

module Dune = Dune_stanza_scan

let sentinel = "ocannl-lifecycle:"
let instrumentation_modules = [ "resource_fault_injection"; "alloc_census" ]

let runnable stanza =
  match Dune.head stanza with
  | Some ("test" | "tests" | "executable" | "executables") -> true
  | Some "library" -> Option.is_some (Dune.field stanza "inline_tests")
  | _ -> false

let libraries stanza =
  match Dune.field stanza "libraries" with
  | None -> Ok []
  | Some terms ->
      List.fold_result terms ~init:[] ~f:(fun names -> function
        | Sexp.Atom name when not (String.is_substring name ~substring:"%{") -> Ok (name :: names)
        | term -> Error ("unsupported lifecycle library dependency: " ^ Sexp.to_string term))

(* Owner identity comes from the real Dune library's modules, not its current public name. Following
   declared library dependencies admits [ocannl] without a copied capability list. *)
let capability files stanza =
  let declarations =
    List.concat_map files ~f:(fun (path, content) ->
        Dune.walk "" (Dune.stanzas content) ~f:(fun subdir library ->
            if not (Option.equal String.equal (Dune.head library) (Some "library")) then []
            else
              let names =
                Dune.names_of library
                @ Option.value_map (Dune.field library "public_name") ~default:[] ~f:(function
                  | [ Sexp.Atom name ] -> [ name ]
                  | _ -> [])
              in
              let owner =
                String.equal path "arrayjit/lib/dune"
                && String.is_empty subdir
                &&
                match Dune.explicit_modules library with
                | Dune.Named modules ->
                    List.for_all instrumentation_modules ~f:(fun name ->
                        List.mem (List.map modules ~f:String.lowercase) name ~equal:String.equal)
                | Dune.Default_less _ -> false
              in
              List.map names ~f:(fun name -> (name, (library, owner)))))
  in
  let rec reaches seen library =
    if Set.mem seen library then Ok false
    else
      match List.filter declarations ~f:(fun (name, _) -> String.equal library name) with
      | [] -> Ok false (* An external library is not the repository's instrumentation owner. *)
      | [ (_, (_, true)) ] -> Ok true
      | [ (_, (stanza, false)) ] -> (
          match libraries stanza with
          | Error _ as error -> error
          | Ok names -> reaches_any (Set.add seen library) names)
      | _ -> Error ("ambiguous lifecycle library name: " ^ library)
  and reaches_any seen names =
    (* A concrete owner path suffices even beside an optional dependency. When no such path exists,
       an unsupported dependency cannot be silently treated as absent. *)
    let answers = List.map names ~f:(reaches seen) in
    if List.exists answers ~f:(function Ok true -> true | _ -> false) then Ok true
    else
      match List.find_map answers ~f:(function Error why -> Some why | Ok _ -> None) with
      | Some why -> Error why
      | None -> Ok false
  in
  match libraries stanza with
  | Error _ as error -> error
  | Ok names -> reaches_any (Set.empty (module String)) names

let contract ~files content =
  Dune.contained_marker_contract content ~sentinel
    ~parse_declaration:(fun _ ~declaration ~reason:_ ->
      let names = String.split declaration ~on:',' |> List.map ~f:String.strip in
      if List.exists names ~f:String.is_empty then Error [ "empty lifecycle unit name" ]
      else if List.contains_dup names ~compare:String.compare then
        Error [ "duplicate lifecycle unit name" ]
      else Ok names)
    ~belongs:(fun marked names ->
      let stanza = marked.Dune.marked_sexp in
      if not (runnable stanza) then
        Error [ "lifecycle markers belong on tests, executables or inline-test libraries" ]
      else if
        List.exists names ~f:(fun name ->
            not (List.mem (Dune.names_of stanza) name ~equal:String.equal))
      then Error [ "lifecycle marker names a unit this stanza does not build" ]
      else
        match capability files stanza with
        | Error why -> Error [ why ]
        | Ok false -> Error [ "lifecycle probe libraries do not reach the instrumentation owner" ]
        | Ok true -> Ok ())

let issues contract =
  let structural =
    List.map contract.Dune.contract_issues ~f:(function
      | Dune.Malformed_marker { issue_line; issue_malformed; _ } ->
          Printf.sprintf "line %d: %s" issue_line
            (Dune.marker_malformed_reason ~sentinel ~separator_subject:"unit names"
               ~grammar:"ocannl-lifecycle: <name>[,<name>...] -- <reason>" issue_malformed)
      | Dune.Marker_outside_stanza { issue_line; _ } ->
          Printf.sprintf "line %d: lifecycle marker outside a stanza" issue_line
      | Dune.Marker_in_wrong_stanza { issue_line; issue_why; _ } ->
          Printf.sprintf "line %d: %s" issue_line issue_why
      | Dune.Marker_outside_comment _ -> "lifecycle marker outside a Dune comment")
  in
  structural
  @ List.filter_map contract.Dune.contract_stanzas ~f:(fun marked ->
      if List.length marked.Dune.stanza_markers > 1 then
        Some
          (Printf.sprintf "line %d: duplicate lifecycle markers on one stanza"
             marked.Dune.marker_stanza.Dune.marked_line)
      else None)

let members contract =
  List.concat_map contract.Dune.contract_stanzas ~f:(fun marked ->
      List.concat_map marked.Dune.stanza_markers ~f:(fun marker ->
          List.map marker.Dune.contained_value ~f:(fun name ->
              ( marked.Dune.marker_stanza.Dune.marked_subdir,
                marked.Dune.marker_stanza.Dune.marked_sexp,
                name ))))
