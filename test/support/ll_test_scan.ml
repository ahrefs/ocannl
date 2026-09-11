open Base
(** A deliberately syntactic ratchet, not a type checker. Constructor names are derived from
    Low_level's t/scalar_t declarations, independent of qualifier spelling (including opens and
    aliases). A same-named foreign constructor is conservatively counted. Expressions and patterns
    are separate: one record construction or one recursive binding inspecting at least two distinct
    IR constructors warrants the harness. Quoted fixtures and comments are not AST nodes. Linking
    the harness is the adoption boundary; this does not ban test-specific match arms. *)

open Ppxlib
module Dune = Dune_stanza_scan

type census = { records : int; traversals : int }
type exemption = Migration of census | Permanent

let constructors source =
  let names = ref [] and records = ref [] in
  Parse.implementation (Lexing.from_string source)
  |> List.iter ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_type (_, declarations) ->
          List.iter declarations ~f:(fun declaration ->
              if List.mem [ "t"; "scalar_t" ] declaration.ptype_name.txt ~equal:String.equal then
                match declaration.ptype_kind with
                | Ptype_variant ctors ->
                    List.iter ctors ~f:(fun c ->
                        names := c.pcd_name.txt :: !names;
                        match c.pcd_args with
                        | Pcstr_record _ -> records := c.pcd_name.txt :: !records
                        | _ -> ())
                | _ -> ())
      | _ -> ());
  (Set.of_list (module String) !names, Set.of_list (module String) !records)

let census ~constructors:(constructors, record_constructors) source =
  let known ident = Set.mem constructors (Longident.last_exn ident) in
  let records = ref 0 and traversals = ref 0 in
  let inspect bindings =
    List.iter bindings ~f:(fun binding ->
        let patterns = Hash_set.create (module String) in
        let walker =
          object (self)
            inherit Ast_traverse.iter as super

            method! structure_item item =
              match item.pstr_desc with
              | Pstr_value (Recursive, _) -> ()
              | _ -> super#structure_item item

            method! expression expr =
              match expr.pexp_desc with
              | Pexp_let (Recursive, _, body) -> self#expression body
              | _ -> super#expression expr

            method! pattern p =
              (match p.ppat_desc with
              | Ppat_construct ({ txt; _ }, _) when known txt ->
                  Hash_set.add patterns (Longident.last_exn txt)
              | _ -> ());
              (* Nested subpatterns matter; nested value definitions have their own census. *)
              super#pattern p
          end
        in
        walker#expression binding.pvb_expr;
        if Hash_set.length patterns >= 2 then Int.incr traversals)
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        (match item.pstr_desc with Pstr_value (Recursive, bindings) -> inspect bindings | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_construct ({ txt; _ }, Some { pexp_desc = Pexp_record _; _ })
          when Set.mem record_constructors (Longident.last_exn txt) ->
            Int.incr records
        | Pexp_let (Recursive, bindings, _) -> inspect bindings
        | _ -> ());
        super#expression expr
    end
  in
  walker#structure (Parse.implementation (Lexing.from_string source));
  { records = !records; traversals = !traversals }

let needs_harness { records; traversals } = records >= 1 || traversals >= 1

let stanza_links ~directory_modules ~module_name ~stanzas stanza =
  Option.value_map (Dune.head stanza) ~default:false ~f:(fun head ->
      List.mem Dune.module_bearing_heads head ~equal:String.equal)
  && List.exists (Dune.modules_of ~directory_modules stanzas stanza) ~f:(fun name ->
      String.Caseless.equal name module_name)
  && Option.value_map (Dune.field stanza "libraries") ~default:false ~f:(fun libraries ->
      List.mem libraries (Sexp.Atom "ll_test") ~equal:Sexp.equal)

let linked ~directory_modules ~module_name stanzas =
  List.exists stanzas ~f:(stanza_links ~directory_modules ~module_name ~stanzas)

(** A select arm is source for the generated target module, never a module named *.real or
    *.missing. Keep the owning stanza with the relationship: another stanza linking ll_test cannot
    cover it, and every owner of a reused arm must adopt the harness. *)
let select_arms stanza =
  Option.value (Dune.field stanza "libraries") ~default:[]
  |> List.concat_map ~f:(function
    | Sexp.List (Sexp.Atom "select" :: Sexp.Atom target :: Sexp.Atom "from" :: arms)
      when String.is_suffix target ~suffix:".ml" ->
        List.filter_map arms ~f:(function
          | Sexp.List terms -> (
              match
                List.drop_while terms ~f:(fun term -> not (Sexp.equal term (Sexp.Atom "->")))
              with
              | [ Sexp.Atom "->"; Sexp.Atom source ] -> Some (target, source)
              | _ -> None)
          | _ -> None)
    | _ -> [])

(** Fold physical dune files and parent subdir blocks into the same directory groups before
    resolving defaults. Directory-spanning ownership modes remain explicit refusals, as in
    env_var_deps; silently treating them as unlinked would misdiagnose an adopted test. *)
let ownership ~sources ~dune_files =
  let groups =
    List.concat_map dune_files ~f:(fun (path, content) ->
        let dir = Stdlib.Filename.dirname path in
        Dune.walk dir (Dune.stanzas content) ~f:(fun dir stanza ->
            let dir = Dune.normalize_path dir in
            [ ((if String.is_empty dir then "." else dir), stanza) ]))
    |> Map.of_alist_multi (module String)
  in
  let problems =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        if
          List.exists sources ~f:(fun path ->
              String.equal dir "." || String.is_prefix path ~prefix:(dir ^ "/"))
        then
          List.filter_map stanzas ~f:(fun stanza ->
              match stanza with
              | Sexp.List [ Sexp.Atom "include_subdirs"; Sexp.Atom "no" ] -> None
              | Sexp.List (Sexp.Atom (("include_subdirs" | "include") as directive) :: _) ->
                  Some (dir ^ ": unsupported module ownership directive " ^ directive)
              | _ -> None)
        else [])
  in
  let selections =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        List.concat_map stanzas ~f:(fun stanza ->
            select_arms stanza
            |> List.map ~f:(fun (target, arm) ->
                (dir, stanza, target, Dune.normalize_path (Dune.in_subdir dir arm)))))
  in
  let module_name path = Stdlib.Filename.remove_extension (Stdlib.Filename.basename path) in
  let directory_modules dir =
    List.filter_map sources ~f:(fun source ->
        if
          String.equal (Stdlib.Filename.dirname source) dir
          && not (List.exists selections ~f:(fun (_, _, _, arm) -> String.equal source arm))
        then Some (module_name source)
        else None)
    @ List.filter_map selections ~f:(fun (owner_dir, _, target, _) ->
        Option.some_if (String.equal owner_dir dir) (module_name target))
    |> List.dedup_and_sort ~compare:String.compare
  in
  let stanzas_at dir = Option.value (Map.find groups dir) ~default:[] in
  let is_linked path =
    match List.filter selections ~f:(fun (_, _, _, arm) -> String.equal path arm) with
    | [] ->
        let dir = Stdlib.Filename.dirname path in
        linked ~directory_modules:(directory_modules dir) ~module_name:(module_name path)
          (stanzas_at dir)
    | owners ->
        List.for_all owners ~f:(fun (dir, stanza, target, _) ->
            stanza_links ~directory_modules:(directory_modules dir)
              ~module_name:(module_name target) ~stanzas:(stanzas_at dir) stanza)
  in
  (is_linked, problems)

let test_source path =
  (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/")
  && String.is_suffix path ~suffix:".ml"

let violations ~exemptions rows =
  let debt = List.filter rows ~f:(fun (_, counts, linked) -> needs_harness counts && not linked) in
  let missing =
    List.filter_map debt ~f:(fun (path, counts, _) ->
        match List.find exemptions ~f:(fun (name, _, _) -> String.equal name path) with
        | None -> Some (path ^ ": requires ll_test or a named migration exemption")
        | Some (_, Permanent, _) -> None
        | Some (_, Migration cap, _) ->
            if counts.records <= cap.records && counts.traversals <= cap.traversals then None
            else
              Some
                (Printf.sprintf
                   "%s: migration debt grew beyond ll_test baseline (records %d/%d, traversals \
                    %d/%d)"
                   path counts.records cap.records counts.traversals cap.traversals))
  in
  let stale =
    List.filter_map exemptions ~f:(fun (path, _, _) ->
        if List.exists debt ~f:(fun (name, _, _) -> String.equal name path) then None
        else Some (path ^ ": stale ll_test exemption"))
  in
  missing @ stale
