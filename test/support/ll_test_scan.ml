open Base
(** A deliberately syntactic ratchet, not a type checker. Constructor names are derived from
    Low_level's t/scalar_t declarations, independent of qualifier spelling (including opens and
    aliases). A same-named foreign constructor is conservatively counted. Expressions and patterns
    are separate: one record construction or one recursive binding inspecting at least two distinct
    IR constructors warrants the harness. Quoted fixtures and comments are not AST nodes. Adoption
    is linking the harness AND calling into its IR surface ({!surface}); linking it for an operand
    helper alone adopts nothing (gh-ocannl-1052). This does not ban test-specific match arms. *)

open Ppxlib
module Dune = Dune_stanza_scan
module Text = Codegen_text_scan

type census = { records : int; traversals : int }
type exemption = Migration of census | Permanent

(** How far a test with detected debt has come: its stanzas do not link the harness, they link it
    but the source calls nothing of its IR surface (the gh-ocannl-1052 state: linked for
    [Ll_test.cycle] alone), or it links the harness and uses it. Only the last is adoption. *)
type adoption = Unlinked | Linked_unused | Adopted

(** The harness, as each module's name and its source relative to the workspace root, in dependency
    order ([Ll_test] includes [Ll_builders]). The names are also how a test reaches the harness's
    values — a qualifier, alias, [open] or [include] of one — beside the libraries {!links_harness}
    reads. *)
let harness_sources =
  [ ("Ll_builders", "test/support/ll_builders.ml"); ("Ll_test", "test/support/ll_test.ml") ]

type surface = bool Map.M(String).t Map.M(String).t
(** Per harness module, every value it exports mapped to whether it belongs to the IR surface. Kept
    per module because the two can disagree on a name: [Ll_test] includes [Ll_builders] and may
    redefine what it included, and the later definition is the one [Ll_test.name] calls. *)

(** [members surface module_name ~ir] is the sorted names [module_name] exports inside ([~ir:true])
    or outside the IR surface. *)
let members (surface : surface) module_name ~ir =
  Option.value_map (Map.find surface module_name) ~default:[] ~f:(fun values ->
      Map.keys (Map.filter values ~f:(Bool.equal ir)))

let is_ir (surface : surface) module_name name =
  Option.value ~default:false
    (Option.bind (Map.find surface module_name) ~f:(fun values -> Map.find values name))

let rec head = function Longident.Lident s -> s | Ldot (p, _) | Lapply (p, _) -> head p

let vars_in iterate =
  let found = Hash_set.create (module String) in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with Ppat_var { txt; _ } -> Hash_set.add found txt | _ -> ());
        super#pattern p
    end
  in
  iterate collector;
  found

(** [surface sources] classifies each harness module's top-level values, DERIVED from their
    definitions rather than listed, so the next helper added to [Ll_test] lands in the right class
    without anyone deciding it: a value is IR when its definition mentions a module path rooted at
    [Ir] (or an alias of one, [module LL = Ir.Low_level]) anywhere — body, type annotation, record
    field — or calls an IR value: unqualified, as the definition in scope at that point (the
    module's own earlier definitions, and what it [include]d or [open]ed from a harness module
    before), unless a local binding shadows it ([cycle_flat]'s [show set]); or qualified through a
    harness module classified before it, or an alias of one ([Ll_builders.seq]). That puts every
    builder, traversal and pipeline helper in the surface ([tick] only through [add]/[c]/[embed]),
    and leaves the operand-data helpers ([cycle], [cycle_flat], [weighted], [drift], [flat]) and the
    value checks ([blank], [close], [same]) outside: integer and float arithmetic with no IR in it.
    A later definition of a name replaces the earlier one's class, as it replaces its meaning. A
    group of recursive bindings is classified together. *)
let surface sources : surface =
  (* Each module's [Ir] aliases travel with it: [Ll_test] reads [LL] through [include
     Ll_builders]. *)
  let ir_aliases = Hashtbl.create (module String) in
  List.fold sources
    ~init:(Map.empty (module String))
    ~f:(fun (classified : surface) (module_name, source) ->
      let ir_modules = Hash_set.of_list (module String) [ "Ir" ] in
      let harness_alias = Hashtbl.create (module String) in
      Map.iter_keys classified ~f:(fun m -> Hashtbl.set harness_alias ~key:m ~data:m);
      let harness_of module_expr =
        Option.bind (Text.module_source_name module_expr) ~f:(Hashtbl.find harness_alias)
      in
      let exported = ref (Map.empty (module String))
      and visible = ref (Map.empty (module String)) in
      let bring_in ?(export = true) module_expr =
        Option.iter (harness_of module_expr) ~f:(fun h ->
            List.iter (Hashtbl.find_multi ir_aliases h) ~f:(Hash_set.add ir_modules);
            let values = Map.find_exn classified h in
            let over into = Map.merge_skewed into values ~combine:(fun ~key:_ _ later -> later) in
            visible := over !visible;
            if export then exported := over !exported)
      in
      List.iter (Text.structure_of source) ~f:(fun item ->
          match item.pstr_desc with
          | Pstr_module { pmb_name = { txt = Some alias; _ }; pmb_expr; _ } -> (
              (match pmb_expr.pmod_desc with
              | Pmod_ident { txt; _ } when Hash_set.mem ir_modules (head txt) ->
                  Hash_set.add ir_modules alias
              | _ -> ());
              match harness_of pmb_expr with
              | Some h -> Hashtbl.set harness_alias ~key:alias ~data:h
              | None -> Hashtbl.remove harness_alias alias)
          | Pstr_include { pincl_mod; _ } -> bring_in pincl_mod
          | Pstr_open { popen_expr; _ } -> bring_in ~export:false popen_expr
          | Pstr_value (_, bindings) ->
              let names = vars_in (fun c -> List.iter bindings ~f:(fun b -> c#pattern b.pvb_pat)) in
              let local =
                vars_in (fun c -> List.iter bindings ~f:(fun b -> c#expression b.pvb_expr))
              in
              let reaches = ref false in
              let walker =
                object
                  inherit Ast_traverse.iter as super
                  method! longident l = if Hash_set.mem ir_modules (head l) then reaches := true

                  method! expression e =
                    (match e.pexp_desc with
                    | Pexp_ident { txt = Lident name; _ } ->
                        if
                          (not (Hash_set.mem local name))
                          && Option.value ~default:false (Map.find !visible name)
                        then reaches := true
                    | Pexp_ident { txt = Ldot (qualifier, name); _ } -> (
                        match Hashtbl.find harness_alias (Longident.last_exn qualifier) with
                        | Some h when is_ir classified h name -> reaches := true
                        | _ -> ())
                    | _ -> ());
                    super#expression e
                end
              in
              List.iter bindings ~f:walker#value_binding;
              Hash_set.iter names ~f:(fun name ->
                  exported := Map.set !exported ~key:name ~data:!reaches;
                  visible := Map.set !visible ~key:name ~data:!reaches)
          | _ -> ());
      Hash_set.iter ir_modules ~f:(fun alias ->
          Hashtbl.add_multi ir_aliases ~key:module_name ~data:alias);
      Map.set classified ~key:module_name ~data:!exported)

type scope = { modules : string Map.M(String).t; opened : string list }
(** What names a harness module in the lexical scope the walk is at: every local module name
    denoting one (the harness's own names to begin with, then aliases, each shadowed by a later
    binding of the same name to anything else), and the harness modules opened or included,
    innermost first. *)

(** [uses_surface ~surface source]: whether the test calls an IR value of the harness — through a
    qualifier naming a harness module or an alias of one ([Ll_test.set], [module B = Ll_builders]
    then [B.loop_n]), or unqualified within the scope of an [open]/[include] of one, judged by the
    class that module gives the name. Scoped the way the language scopes it: a structure-level
    alias, open or include governs the items after it, an expression-level one its body, a nested
    structure's die with it, and a later [module L = ...] rebinds [L]. An unqualified name the file
    binds anywhere for itself is not credited, so a local [set] under an [open Ll_test] is not taken
    for the builder — erring toward reporting debt, the direction the ratchet exists to keep honest.
*)
let uses_surface ~(surface : surface) source =
  let structure = Text.structure_of source in
  let bound = Text.names_bound_anywhere structure in
  let found = ref false in
  let harness_of scope module_expr =
    Option.bind (Text.module_source_name module_expr) ~f:(Map.find scope.modules)
  in
  let bind scope name module_expr =
    {
      scope with
      modules =
        (match harness_of scope module_expr with
        | Some h -> Map.set scope.modules ~key:name ~data:h
        | None -> Map.remove scope.modules name);
    }
  in
  let opening scope module_expr =
    Option.map (harness_of scope module_expr) ~f:(fun h ->
        { scope with opened = h :: scope.opened })
  in
  let exports h name =
    Option.value_map (Map.find surface h) ~default:false ~f:(fun values -> Map.mem values name)
  in
  let walker =
    object (self)
      inherit [scope] Ast_traverse.map_with_context as super

      method! structure scope items =
        ignore
          (List.fold items ~init:scope ~f:(fun scope item ->
               ignore (self#structure_item scope item : structure_item);
               match item.pstr_desc with
               | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
                   bind scope name pmb_expr
               | Pstr_open { popen_expr = m; _ } | Pstr_include { pincl_mod = m; _ } ->
                   Option.value (opening scope m) ~default:scope
               | _ -> scope));
        items

      method! expression scope e =
        match e.pexp_desc with
        | Pexp_open ({ popen_expr; _ }, body) -> (
            match opening scope popen_expr with
            | Some inner ->
                ignore (self#expression inner body : expression);
                e
            | None -> super#expression scope e)
        | Pexp_letmodule ({ txt = Some name; _ }, module_expr, body) ->
            ignore (self#module_expr scope module_expr : module_expr);
            ignore (self#expression (bind scope name module_expr) body : expression);
            e
        | Pexp_ident { txt = Lident name; _ } ->
            (if not (Set.mem bound name) then
               match List.find scope.opened ~f:(fun h -> exports h name) with
               | Some h when is_ir surface h name -> found := true
               | _ -> ());
            e
        | Pexp_ident { txt = Ldot (qualifier, name); _ } ->
            (match Map.find scope.modules (Longident.last_exn qualifier) with
            | Some h when is_ir surface h name -> found := true
            | _ -> ());
            e
        | _ -> super#expression scope e
    end
  in
  let modules =
    Map.of_alist_exn (module String) (List.map (Map.keys surface) ~f:(fun m -> (m, m)))
  in
  ignore (walker#structure { modules; opened = [] } structure : structure);
  !found

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

let stanza_owns ~directory_modules ~module_name ~stanzas stanza =
  Option.value_map (Dune.head stanza) ~default:false ~f:(fun head ->
      List.mem Dune.module_bearing_heads head ~equal:String.equal)
  && List.exists (Dune.modules_of ~directory_modules stanzas stanza) ~f:(fun name ->
      String.Caseless.equal name module_name)

let links_harness stanza =
  Option.value_map (Dune.field stanza "libraries") ~default:false ~f:(fun libraries ->
      List.exists [ "ll_test"; "arrayjit.ll_builders" ] ~f:(fun name ->
          List.mem libraries (Sexp.Atom name) ~equal:Sexp.equal))

let stanza_links ~directory_modules ~module_name ~stanzas stanza =
  stanza_owns ~directory_modules ~module_name ~stanzas stanza && links_harness stanza

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
let test_source path =
  (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/")
  && String.is_suffix path ~suffix:".ml"
  && not
       (String.equal (Stdlib.Filename.dirname path) "test/ppx"
       && String.is_suffix path ~suffix:"_expected.ml")

(** Literal copy forms used by this tree. Do not silently guess at globs, pforms or generated inputs
    outside the declared test corpus. Non-ML copies do not affect module ownership. *)
let copy_input stanza =
  match Dune.head stanza with
  | Some ("copy_files" | "copy_files#") -> (
      let input =
        match stanza with
        | Sexp.List [ _; Sexp.Atom input ] -> Some input
        | _ -> (
            match Dune.field stanza "files" with
            | Some [ Sexp.Atom input ] -> Some input
            | _ -> None)
      in
      match input with
      | None -> Error "unsupported copy_files source form"
      | Some input ->
          let dynamic s = String.exists s ~f:(fun c -> String.mem "*?[]{}%" c) in
          let extension = Stdlib.Filename.extension input in
          if
            (not (String.is_empty extension))
            && (not (dynamic extension))
            && not (String.equal extension ".ml")
          then Ok None
          else if dynamic input then Error "unsupported copy_files glob or dynamic source"
          else Ok (Option.some_if (String.is_suffix input ~suffix:".ml") input))
  | _ -> Ok None

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
  let copy_errors = ref [] in
  let copies =
    Map.to_alist groups
    |> List.concat_map ~f:(fun (dir, stanzas) ->
        List.filter_map stanzas ~f:(fun stanza ->
            match copy_input stanza with
            | Error reason ->
                if test_source (Dune.normalize_path (Dune.in_subdir dir "probe.ml")) then
                  copy_errors := (dir ^ ": " ^ reason) :: !copy_errors;
                None
            | Ok None -> None
            | Ok (Some input) ->
                Some
                  ( Dune.normalize_path (Dune.in_subdir dir input),
                    Dune.normalize_path (Dune.in_subdir dir (Stdlib.Filename.basename input)) )))
  in
  let inputs =
    copies
    @ List.map selections ~f:(fun (dir, _, target, arm) ->
        (arm, Dune.normalize_path (Dune.in_subdir dir target)))
  in
  let rec declared_origin seen path =
    if List.mem sources path ~equal:String.equal then true
    else if List.mem seen path ~equal:String.equal then false
    else
      let origins =
        List.filter_map inputs ~f:(fun (input, target) ->
            Option.some_if (String.equal target path) input)
      in
      (not (List.is_empty origins)) && List.for_all origins ~f:(declared_origin (path :: seen))
  in
  let input_errors =
    List.filter_map inputs ~f:(fun (input, target) ->
        if test_source target && not (declared_origin [] input) then
          Some (target ^ ": source input outside declared test corpus: " ^ input)
        else None)
  in
  let module_name path = Stdlib.Filename.remove_extension (Stdlib.Filename.basename path) in
  let directory_modules dir =
    List.filter_map
      (sources @ List.map copies ~f:snd)
      ~f:(fun source ->
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
  let rec owners seen path =
    if List.mem seen path ~equal:String.equal then [ false ]
    else
      let seen = path :: seen in
      let direct =
        match List.filter selections ~f:(fun (_, _, _, arm) -> String.equal path arm) with
        | [] ->
            if not (test_source path) then []
            else
              let dir = Stdlib.Filename.dirname path in
              List.filter (stanzas_at dir)
                ~f:
                  (stanza_owns ~directory_modules:(directory_modules dir)
                     ~module_name:(module_name path) ~stanzas:(stanzas_at dir))
              |> List.map ~f:links_harness
        | selected ->
            List.concat_map selected ~f:(fun (dir, stanza, target, _) ->
                stanza_links ~directory_modules:(directory_modules dir)
                  ~module_name:(module_name target) ~stanzas:(stanzas_at dir) stanza
                :: copied_owners seen (Dune.normalize_path (Dune.in_subdir dir target)))
      in
      direct @ copied_owners seen path
  and copied_owners seen path =
    List.filter copies ~f:(fun (source, _) -> String.equal source path)
    |> List.concat_map ~f:(fun (_, target) -> owners seen target)
  in
  let is_linked path =
    match owners [] path with [] -> false | owners -> List.for_all owners ~f:Fn.id
  in
  (is_linked, problems @ List.rev !copy_errors @ input_errors)

let violations ~exemptions rows =
  let debt =
    List.filter rows ~f:(fun (_, counts, adoption) ->
        needs_harness counts && match adoption with Adopted -> false | _ -> true)
  in
  let missing =
    List.filter_map debt ~f:(fun (path, counts, adoption) ->
        match List.find exemptions ~f:(fun (name, _, _) -> String.equal name path) with
        | None -> (
            match adoption with
            | Linked_unused ->
                Some
                  (path
                 ^ ": links ll_test but calls none of its IR surface, so its hand-built Low_level \
                    is still debt; use the harness's builders or name a migration exemption")
            | Unlinked | Adopted -> Some (path ^ ": requires ll_test or a named migration exemption")
            )
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
