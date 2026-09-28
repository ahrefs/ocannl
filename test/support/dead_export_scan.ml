(** The source reader behind the dead implicit-export ratchet.

    This is deliberately a first cut. An implementation without an [.mli] exports every
    source-declared top-level [let] and [external], including helpers intended only for the
    implementation. Source-declared [let] values whose name starts with [_] are excluded: the author
    already marked them deliberately unused, matching OCaml warning 32. This also covers patterns
    and extension payloads; externals and inferred deriving names retain their existing census
    policy, including polymorphic-variant parser helpers. We enumerate those declarations in
    [arrayjit/lib/], [tensor/] and [lib/], then count references from every other OCaml source.
    Top-level type declarations are a second, coarser census with its own section below.

    A reference is conservative: a direct qualified path ([M.v]), a path through a module alias, or
    an unqualified identifier inside the lexical range of [open M]. Alias scopes are deliberately
    over-approximated to the whole source. An [include M] is not a use of anything (gh-ocannl-1085):
    it makes the including module another receiver of [M]'s values, so [N.v] after
    [module N = struct include M end] -- or [Utils.insert] after [utils.ml]'s [include Datatypes] --
    counts, transitively and through aliases, as does a bare [v] where [N] is opened or below the
    include itself; a value the include re-exports but nobody spells reads dead. Receivers match by
    their last qualifier, and an includer that shadows an included value still credits it. These
    choices can hide a dead export through a false positive, but cannot falsely reject an ordinary
    use, save through an include this reader does not follow: of a functor application, or inside a
    functor body. Values generated for top-level types by [of_sexp], [compare], and [equal]
    derivings are included (a standalone [of_sexp] deriving as much as the [of_sexp] half of
    [sexp]); their expression extensions count as references without needing to spell the generated
    value. The [sexp_of] converters -- derived or hand-written, recognized by the [sexp_of_] name
    prefix -- are excluded by policy: they are the entry point for debugging and observability, and
    consumed as often through [ppx_minidebug]'s typed log annotations, which expand to converter
    calls this source-level census cannot see, as through spelled references. A [sexp_of] with no
    caller costs nothing and cannot drift from its type, while removing it to satisfy a ratchet only
    takes the converter away from the next debugging session. Values introduced by other PPX
    expansions or by an [include] of another module inside the defining module remain outside this
    source-level census. A bare [include struct ... end] declares into the module and is read like
    top-level items; a constrained [include (struct ... end : S)] carries its own interface, which
    publishes deliberately, so like a module with an [.mli] it is not censused. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Read = Config_key_scan

type export = { module_name : string; value : string; source : string; line : int }

type reference = {
  module_name : string;
  value : string;
  source : string;
  line : int;
  spelling : string;
}

let export_key ({ module_name; value; _ } : export) = module_name ^ "." ^ value

let valid_module_stem stem =
  (not (String.is_empty stem))
  && Char.is_alpha stem.[0]
  && String.for_all stem ~f:(fun c -> Char.is_alphanum c || Char.equal c '_' || Char.equal c '\'')

(** The OCaml module name of a direct [*.ml] source. Dune select alternatives such as
    [cuda_backend_impl.cudajit.ml] are not modules under that basename and return [None]. *)
let module_name_of_source source =
  match String.chop_suffix (Stdlib.Filename.basename source) ~suffix:".ml" with
  | Some stem when valid_module_stem stem -> Some (String.capitalize stem)
  | Some _ | None -> None

let pattern_names pattern =
  let names = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_var { txt; _ } -> names := txt :: !names | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  List.dedup_and_sort !names ~compare:String.compare

let deriver_of_expression expression =
  let rec head expression =
    match expression.pexp_desc with
    | Pexp_ident { txt; _ } -> Read.flatten_longident txt |> List.last
    | Pexp_apply (function_, _) | Pexp_constraint (function_, _) -> head function_
    | _ -> None
  in
  match expression.pexp_desc with
  | Pexp_tuple expressions -> List.filter_map expressions ~f:head
  | _ -> Option.to_list (head expression)

let derivers_of_type_declaration declaration =
  List.concat_map declaration.ptype_attributes ~f:(fun attribute ->
      match (attribute.attr_name.txt, attribute.attr_payload) with
      | "deriving", PStr items ->
          List.concat_map items ~f:(fun item ->
              match item.pstr_desc with
              | Pstr_eval (expression, _) -> deriver_of_expression expression
              | _ -> [])
      | _ -> [])

let comparison_name deriver type_name =
  if String.equal type_name "t" then deriver else deriver ^ "_" ^ type_name

let is_polymorphic_variant declaration =
  match declaration.ptype_manifest with
  | Some { ptyp_desc = Ptyp_variant _; _ } -> true
  | Some _ | None -> false

let of_sexp_names declaration =
  let type_name = declaration.ptype_name.txt in
  let public = type_name ^ "_of_sexp" in
  if is_polymorphic_variant declaration then [ public; "__" ^ public ^ "__" ] else [ public ]

let derived_names ~derivers declaration =
  let type_name = declaration.ptype_name.txt in
  List.concat_map derivers ~f:(function
    | "of_sexp" | "sexp" -> of_sexp_names declaration
    | "compare" -> [ comparison_name "compare" type_name ]
    | "equal" -> [ comparison_name "equal" type_name ]
    | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

let is_sexp_of_converter value = String.is_prefix value ~prefix:"sexp_of_"

let exports_of_source ~source contents =
  match module_name_of_source source with
  | None -> []
  | Some module_name ->
      let rec items acc structure = List.fold structure ~init:acc ~f:item
      and item acc structure_item =
        match structure_item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.fold bindings ~init:acc ~f:(fun acc binding ->
                List.fold (pattern_names binding.pvb_pat) ~init:acc ~f:(fun acc value ->
                    if String.is_prefix value ~prefix:"_" || is_sexp_of_converter value then acc
                    else
                      { module_name; value; source; line = binding.pvb_loc.loc_start.pos_lnum }
                      :: acc))
        | Pstr_primitive description ->
            {
              module_name;
              value = description.pval_name.txt;
              source;
              line = description.pval_loc.loc_start.pos_lnum;
            }
            :: acc
        | Pstr_type (_, declarations) ->
            (* The parser attaches a group's trailing [@@deriving] to its last declaration, while
               the deriver applies it to every declaration in the mutually recursive group. *)
            let derivers = List.concat_map declarations ~f:derivers_of_type_declaration in
            List.fold declarations ~init:acc ~f:(fun acc declaration ->
                List.fold (derived_names ~derivers declaration) ~init:acc ~f:(fun acc value ->
                    { module_name; value; source; line = declaration.ptype_loc.loc_start.pos_lnum }
                    :: acc))
        (* [let%foo] is an extension carrying a structure payload before the PPX rewrites it. It is
           still a source-declared top-level value, so unwrap structure payloads at this level but
           never descend into a nested module. *)
        | Pstr_extension ((_, PStr nested), _) -> items acc nested
        (* [include struct ... end] declares into the enclosing module as much as a bare item. *)
        | Pstr_include { pincl_mod = { pmod_desc = Pmod_structure nested; _ }; _ } ->
            items acc nested
        | _ -> acc
      in
      items [] (Read.structure_of contents)
      |> List.dedup_and_sort ~compare:(fun (a : export) (b : export) ->
          match String.compare a.value b.value with
          | 0 -> String.compare a.source b.source
          | ordering -> ordering)

let path_last path = List.last path

let path_qualifier path =
  match List.rev path with _value :: qualifier :: _ -> Some qualifier | _ -> None

let flattened_longident longident = try Some (Read.flatten_longident longident) with _ -> None

let module_expr_name module_expr =
  let rec unwrap module_expr =
    match module_expr.pmod_desc with Pmod_constraint (inner, _) -> unwrap inner | _ -> module_expr
  in
  match (unwrap module_expr).pmod_desc with
  | Pmod_ident { txt; _ } -> ( try Read.flatten_longident txt |> List.last with _ -> None)
  | _ -> None

let extension_deriver = function
  | "equal" -> Some "equal"
  | "compare" -> Some "compare"
  | "of_sexp" -> Some "of_sexp"
  | _ -> None

let derived_name_for_extension deriver type_name =
  match deriver with
  | "equal" | "compare" -> comparison_name deriver type_name
  | "of_sexp" -> type_name ^ "_of_sexp"
  | _ -> assert false

let reference_derivers = function
  | "sexp" | "of_sexp" -> [ "of_sexp" ]
  | ("compare" | "equal") as deriver -> [ deriver ]
  | _ -> []

let has_attribute name attributes =
  List.exists attributes ~f:(fun attribute -> String.equal attribute.attr_name.txt name)

let opaque_sexp_wrapper core_type =
  match core_type.ptyp_desc with
  | Ptyp_constr ({ txt; _ }, [ _ ]) -> (
      match flattened_longident txt with
      | Some path ->
          Option.value_map (path_last path) ~default:false ~f:(String.equal "sexp_opaque")
      | None -> false)
  | _ -> false

let core_type_ignored ~deriver core_type =
  match deriver with
  | "of_sexp" ->
      has_attribute "sexp.opaque" core_type.ptyp_attributes
      || has_attribute "sexp.ignore" core_type.ptyp_attributes
      || opaque_sexp_wrapper core_type
  | "compare" -> has_attribute "compare.ignore" core_type.ptyp_attributes
  | "equal" -> has_attribute "equal.ignore" core_type.ptyp_attributes
  | _ -> false

let label_ignored ~deriver label =
  match deriver with
  | "of_sexp" -> has_attribute "sexp.ignore" label.pld_attributes
  | "compare" -> has_attribute "compare.ignore" label.pld_attributes
  | "equal" -> has_attribute "equal.ignore" label.pld_attributes
  | _ -> false

(** The name a top-level item of [source] is reached under: its module name, also for a dune select
    alternative, which implements the module named before its first dot
    ([cuda_backend_impl.cudajit.ml] is [Cuda_backend_impl]). *)
let includer_name_of_source source =
  match String.lsplit2 (Stdlib.Filename.basename source) ~on:'.' with
  | Some (stem, _) when valid_module_stem stem -> Some (String.capitalize stem)
  | Some _ | None -> None

(** Every [include] of a named module in [structure], as [(included, includer)] pairs: the last
    component of the included path, and the name under which the including module is reached -- the
    source's own module at top level, the innermost [module N = struct ... end] around a nested one.
    An [include struct ... end] declares into the module it sits in; an included functor application
    or functor body is not followed. *)
let includes_of ~top structure =
  let found = ref [] in
  let rec items includer structure = List.iter structure ~f:(item includer)
  and item includer structure_item =
    match structure_item.pstr_desc with
    | Pstr_include { pincl_mod = { pmod_desc = Pmod_structure nested; _ }; _ } ->
        items includer nested
    | Pstr_include { pincl_mod; _ } -> (
        match (includer, module_expr_name pincl_mod) with
        | Some includer, Some included -> found := (included, includer) :: !found
        | _ -> ())
    | Pstr_module binding -> module_binding binding
    | Pstr_recmodule bindings -> List.iter bindings ~f:module_binding
    | Pstr_extension ((_, PStr nested), _) -> items includer nested
    | _ -> ()
  and module_binding { pmb_name = { txt; _ }; pmb_expr; _ } =
    let rec body module_expr =
      match module_expr.pmod_desc with
      | Pmod_structure nested -> items txt nested
      | Pmod_constraint (inner, _) -> body inner
      | _ -> ()
    in
    body pmb_expr
  in
  items top structure;
  !found

(** For each of [modules], the names its values are reached under in [parsed] sources: itself, and
    every module that includes it or one of those, through a local alias as much as by name. A name,
    like every receiver here, is matched as the last qualifier of a path, so a same-named module
    elsewhere credits too -- the over-reading direction. *)
let receiver_names ~modules ~parsed =
  let includes =
    List.filter_map parsed ~f:(fun (source, structure) ->
        match includes_of ~top:(includer_name_of_source source) structure with
        | [] -> None
        | includes -> Some (structure, includes))
  in
  List.map modules ~f:(fun module_name ->
      let rec close names =
        let grown =
          List.fold includes ~init:names ~f:(fun names (structure, includes) ->
              let local =
                Set.fold names ~init:names ~f:(fun local name ->
                    let aliases, _opened, _ranges =
                      Read.module_bindings_of structure ~wanted:name
                    in
                    Set.union local (Set.of_list (module String) aliases))
              in
              List.fold includes ~init:names ~f:(fun names (included, includer) ->
                  if Set.mem local included then Set.add names includer else names))
        in
        if Set.length grown = Set.length names then names else close grown
      in
      (module_name, Set.to_list (close (Set.singleton (module String) module_name))))
  |> Map.of_alist_exn (module String)

(** References to [exports] from [sources]. Sources are [(repository-relative path, contents)]. *)
let references ~(exports : export list) ~sources =
  let modules =
    List.map exports ~f:(fun export -> export.module_name)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let parsed =
    List.map sources ~f:(fun (source, contents) -> (source, Read.structure_of contents))
  in
  let receiver_names = receiver_names ~modules ~parsed in
  let export_names =
    List.fold exports
      ~init:(Map.empty (module String))
      ~f:(fun map export ->
        Map.update map export.module_name ~f:(function
          | None -> Set.singleton (module String) export.value
          | Some values -> Set.add values export.value))
  in
  List.concat_map parsed ~f:(fun (source, structure) ->
      List.concat_map modules ~f:(fun module_name ->
          let receivers, open_ranges =
            List.fold
              (Map.find_exn receiver_names module_name)
              ~init:(Set.empty (module String), [])
              ~f:(fun (receivers, open_ranges) name ->
                let aliases, _opened, ranges = Read.module_bindings_of structure ~wanted:name in
                ( Set.union receivers (Set.of_list (module String) (name :: aliases)),
                  ranges @ open_ranges ))
          in
          let names = Map.find_exn export_names module_name in
          let found = ref [] in
          let add ~value ~line ~spelling =
            if Set.mem names value then
              found := { module_name; value; source; line; spelling } :: !found
          in
          let add_type_references ~deriver ~internal_inherited ~line ~spelling core_type =
            let types =
              object (self)
                inherit Ast_traverse.iter as super

                method private add_path ~internal_of_sexp core_type =
                  match core_type.ptyp_desc with
                  | Ptyp_constr ({ txt; _ }, _) -> (
                      match flattened_longident txt with
                      | None -> ()
                      | Some path -> (
                          let value type_name =
                            if internal_of_sexp then "__" ^ type_name ^ "_of_sexp__"
                            else derived_name_for_extension deriver type_name
                          in
                          match (path_last path, path_qualifier path) with
                          | Some type_name, Some receiver when Set.mem receivers receiver ->
                              add ~value:(value type_name) ~line ~spelling:(spelling path)
                          | Some type_name, None
                            when Read.within open_ranges core_type.ptyp_loc.loc_start.pos_cnum ->
                              add ~value:(value type_name) ~line ~spelling:(spelling [ type_name ])
                          | _ -> ()))
                  | _ -> ()

                method! core_type core_type =
                  if core_type_ignored ~deriver core_type then ()
                  else (
                    self#add_path ~internal_of_sexp:false core_type;
                    super#core_type core_type)

                method! row_field row_field =
                  match (deriver, internal_inherited, row_field.prf_desc) with
                  | "of_sexp", true, Rinherit inherited ->
                      if core_type_ignored ~deriver inherited then ()
                      else (
                        self#add_path ~internal_of_sexp:true inherited;
                        match inherited.ptyp_desc with
                        | Ptyp_constr (_, arguments) -> List.iter arguments ~f:self#core_type
                        | _ -> self#core_type inherited)
                  | _ -> super#row_field row_field

                method! label_declaration label =
                  if label_ignored ~deriver label then () else super#label_declaration label
              end
            in
            types#core_type core_type
          in
          let add_declaration_references ~deriver declaration =
            let line = declaration.ptype_loc.loc_start.pos_lnum in
            let spelling path =
              "[@@deriving " ^ deriver ^ "] over " ^ String.concat ~sep:"." path
            in
            let add_type = add_type_references ~deriver ~internal_inherited:true ~line ~spelling in
            let add_label label =
              if not (label_ignored ~deriver label) then add_type label.pld_type
            in
            match declaration.ptype_kind with
            | Ptype_abstract -> Option.iter declaration.ptype_manifest ~f:add_type
            | Ptype_variant constructors ->
                List.iter constructors ~f:(fun constructor ->
                    match constructor.pcd_args with
                    | Pcstr_tuple types -> List.iter types ~f:add_type
                    | Pcstr_record labels -> List.iter labels ~f:add_label)
            | Ptype_record labels -> List.iter labels ~f:add_label
            | Ptype_open -> ()
          in
          let iterator =
            object
              inherit Ast_traverse.iter as super

              method! structure_item item =
                (* An [include M] is not itself a use: it makes the including module a receiver of
                   M's values ({!receiver_names}), and a use is a spelling through either. *)
                (match item.pstr_desc with
                | Pstr_type (_, declarations) ->
                    let derivers =
                      List.concat_map declarations ~f:derivers_of_type_declaration
                      |> List.concat_map ~f:reference_derivers
                      |> List.dedup_and_sort ~compare:String.compare
                    in
                    List.iter derivers ~f:(fun deriver ->
                        List.iter declarations ~f:(add_declaration_references ~deriver))
                | _ -> ());
                super#structure_item item

              method! attribute attribute =
                if String.equal attribute.attr_name.txt "deriving" then ()
                else super#attribute attribute

              method! expression expression =
                (match Read.longident_of expression with
                | Some path -> (
                    match (path_last path, path_qualifier path) with
                    | Some value, Some receiver when Set.mem receivers receiver ->
                        add ~value ~line:expression.pexp_loc.loc_start.pos_lnum
                          ~spelling:(String.concat ~sep:"." path)
                    | Some value, None
                      when Read.within open_ranges expression.pexp_loc.loc_start.pos_cnum ->
                        add ~value ~line:expression.pexp_loc.loc_start.pos_lnum ~spelling:value
                    | _ -> ())
                | None -> ());
                (match expression.pexp_desc with
                | Pexp_extension ({ txt; _ }, PTyp core_type) -> (
                    match extension_deriver txt with
                    | None -> ()
                    | Some deriver ->
                        add_type_references ~deriver
                          ~internal_inherited:(String.equal deriver "of_sexp")
                          ~line:expression.pexp_loc.loc_start.pos_lnum
                          ~spelling:(fun path ->
                            "[%" ^ deriver ^ ": " ^ String.concat ~sep:"." path ^ "]")
                          core_type)
                | _ -> ());
                super#expression expression
            end
          in
          iterator#structure structure;
          List.rev !found))
  |> List.filter ~f:(fun reference ->
      not
        (List.exists exports ~f:(fun export ->
             String.equal export.module_name reference.module_name
             && String.equal export.source reference.source)))

let counts ~(exports : export list) references =
  let table = Hashtbl.create (module String) in
  List.iter exports ~f:(fun export -> Hashtbl.set table ~key:(export_key export) ~data:0);
  List.iter references ~f:(fun reference ->
      Hashtbl.update table
        (reference.module_name ^ "." ^ reference.value)
        ~f:(fun count -> 1 + Option.value count ~default:0));
  table

(** {1 Type declarations}

    The type census answers one conservative question: does a top-level type's NAME occur anywhere
    in the tree beyond its own declaration? It is a name count, not a resolution. A mention is the
    last component of any type path ([foo], [M.foo], [N.foo] all credit every censused [foo], so a
    same-named type elsewhere hides a dead one), in implementations and interfaces alike, including
    the defining source (a type its own module uses is live, if not public), but excluding the
    declaration's own span, so a recursive type does not credit itself. Package constraints
    ([(module S with type foo = int)]), [with type] constraints, type extensions and [#foo] patterns
    are type paths too. A type whose deriving generates values or modules is also mentioned by a
    spelling of one of them in a value or module path -- never in a label path, and never inside a
    [[@@deriving]] payload, which names derivers rather than using their output -- since a caller
    can use the type through its converter alone. Comments, docstrings and string literals never
    parse into a path, so prose cannot keep a type alive. A type named with a leading [_] is not
    censused, as it is not for OCaml's own unused-type warning (34): its author marked it
    deliberately unused, the convention the value census follows for [let]. As in the value census,
    an [include M] credits none of [M]'s types, and here needs no receiver either: the count is
    blind to qualifiers, so a use through the including module ([N.foo]) is already a mention.

    Out of scope, by design: the constructors and record labels of a type are not resolved to it (a
    record built only by its labels, never annotated, reads as unmentioned), and neither are the
    label-named accessors [[@@deriving fields]] generates -- crediting a label's name would let any
    same-named label elsewhere keep a dead record alive; nor are values, exceptions, module types or
    classes; nor anything a PPX other than the modelled derivings generates. *)

type type_export = {
  module_name : string;
  type_name : string;
  source : string;
  line : int;
  span : int * int;
  mentioned_by : string list;
}

let type_export_key ({ module_name; type_name; _ } : type_export) = module_name ^ "." ^ type_name

(** The values and modules a deriving generates for [declaration], any spelling of which mentions
    the type: the value census's {!derived_names}, plus what that census leaves out -- the [sexp_of]
    converters and the [hash], [enumerate], [variants] and [fields] outputs. The label-named
    accessors of [fields] are left out, by the label boundary above. The synthetic cases hold this
    set equal to the derivers' own expansion. *)
let deriving_mentions ~derivers declaration =
  let type_name = declaration.ptype_name.txt in
  let named ~t ~prefix = if String.equal type_name "t" then t else prefix ^ type_name in
  derived_names ~derivers declaration
  @ List.concat_map derivers ~f:(function
    | "sexp" | "sexp_of" -> [ "sexp_of_" ^ type_name ]
    | "hash" ->
        (* The [hash_<type>] shorthand exists only for a type without parameters. *)
        ("hash_fold_" ^ type_name)
        ::
        (if List.is_empty declaration.ptype_params then [ named ~t:"hash" ~prefix:"hash_" ] else [])
    | "enumerate" -> [ named ~t:"all" ~prefix:"all_of_" ]
    | "variants" -> [ named ~t:"Variants" ~prefix:"Variants_of_" ]
    | "fields" -> [ named ~t:"Fields" ~prefix:"Fields_of_" ]
    | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

let type_exports_of_source ~source contents =
  match module_name_of_source source with
  | None -> []
  | Some module_name ->
      let rec items acc structure = List.fold structure ~init:acc ~f:item
      and item acc structure_item =
        match structure_item.pstr_desc with
        | Pstr_type (_, declarations) ->
            let derivers = List.concat_map declarations ~f:derivers_of_type_declaration in
            List.fold declarations ~init:acc ~f:(fun acc declaration ->
                let type_name = declaration.ptype_name.txt in
                if String.is_prefix type_name ~prefix:"_" then acc
                else
                  let loc = declaration.ptype_loc in
                  {
                    module_name;
                    type_name;
                    source;
                    line = loc.loc_start.pos_lnum;
                    span = (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum);
                    mentioned_by = deriving_mentions ~derivers declaration;
                  }
                  :: acc)
        | Pstr_extension ((_, PStr nested), _) -> items acc nested
        (* [include struct ... end] declares into the enclosing module as much as a bare item. *)
        | Pstr_include { pincl_mod = { pmod_desc = Pmod_structure nested; _ }; _ } ->
            items acc nested
        | _ -> acc
      in
      items [] (Read.structure_of contents)
      |> List.sort ~compare:(fun (a : type_export) (b : type_export) ->
          String.compare (type_export_key a) (type_export_key b))

(** The interface sources among [paths], the [*.mli] counterpart of [Config_key_scan.sources_among]:
    dune's [X.pp.mli] is a preprocessed binary AST beside the [X.mli] it came from, not a source. *)
let interfaces_among paths =
  let interfaces =
    List.filter paths ~f:(String.is_suffix ~suffix:".mli")
    |> List.dedup_and_sort ~compare:String.compare
  in
  let present = Set.of_list (module String) interfaces in
  List.filter interfaces ~f:(fun path ->
      match String.chop_suffix path ~suffix:".pp.mli" with
      | Some stem -> not (Set.mem present (stem ^ ".mli"))
      | None -> true)

(** How many mentions each of [type_exports] has, keyed by {!type_export_key}. [implementations] and
    [interfaces] are [(repository-relative path, contents)]; an interface parses as a signature.
    Raises if a source does not parse. *)
let type_mention_counts ~(type_exports : type_export list) ~implementations ~interfaces =
  let type_names =
    Set.of_list (module String) (List.map type_exports ~f:(fun export -> export.type_name))
  in
  let derived_names =
    Set.of_list
      (module String)
      (List.concat_map type_exports ~f:(fun export -> export.mentioned_by))
  in
  (* Every mention carries its place, to be told apart from the declaration's own span. *)
  let type_mentions = Hashtbl.create (module String) in
  let derived_mentions = Hashtbl.create (module String) in
  let walk ~source =
    (* Read through functor applications: [F(X).foo] names [foo] and spells [F] and [X]. *)
    let rec components : Ppxlib.longident -> string list = function
      | Lident name -> [ name ]
      | Ldot (prefix, name) -> components prefix @ [ name ]
      | Lapply (functor_, argument) -> components functor_ @ components argument
    in
    let record_type_path (path : Ppxlib.longident_loc) =
      let last =
        match path.txt with Lident name | Ldot (_, name) -> Some name | Lapply _ -> None
      in
      match last with
      | Some name when Set.mem type_names name ->
          Hashtbl.add_multi type_mentions ~key:name ~data:(source, path.loc.loc_start.pos_cnum)
      | Some _ | None -> ()
    in
    (* A derived value or module is spelled as a whole value or module path, or as a qualifier in
       one ([Fields_of_foo.names]). Label paths are not read at all. *)
    let record_derived_path (path : Ppxlib.longident_loc) =
      List.iter (components path.txt) ~f:(fun name ->
          if Set.mem derived_names name then
            Hashtbl.add_multi derived_mentions ~key:name ~data:(source, path.loc.loc_start.pos_cnum))
    in
    object
      inherit Ast_traverse.iter as super

      method! core_type core_type =
        (match core_type.ptyp_desc with
        | Ptyp_constr (path, _) | Ptyp_class (path, _) -> record_type_path path
        | Ptyp_package (_, constraints) ->
            List.iter constraints ~f:(fun (path, _) -> record_type_path path)
        | _ -> ());
        super#core_type core_type

      (* [#foo] in a pattern matches the rows of the polymorphic variant [foo]. *)
      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_type path -> record_type_path path | _ -> ());
        super#pattern pattern

      method! with_constraint constraint_ =
        (match constraint_ with
        | Pwith_type (path, _) | Pwith_typesubst (path, _) -> record_type_path path
        | Pwith_module (_, path) | Pwith_modsubst (_, path) -> record_derived_path path
        | Pwith_modtype _ | Pwith_modtypesubst _ -> ());
        super#with_constraint constraint_

      method! type_extension extension =
        record_type_path extension.ptyext_path;
        super#type_extension extension

      method! expression expression =
        (match expression.pexp_desc with Pexp_ident path -> record_derived_path path | _ -> ());
        super#expression expression

      method! module_expr module_expr =
        (match module_expr.pmod_desc with Pmod_ident path -> record_derived_path path | _ -> ());
        super#module_expr module_expr

      (* The signature-side module paths: an alias ([module F = M.Fields_of_foo]) and a module type
         path, an [open], and a module substitution. *)
      method! module_type module_type =
        (match module_type.pmty_desc with
        | Pmty_alias path | Pmty_ident path -> record_derived_path path
        | _ -> ());
        super#module_type module_type

      method! signature_item item =
        (match item.psig_desc with
        | Psig_open { popen_expr = path; _ } -> record_derived_path path
        | Psig_modsubst { pms_manifest = path; _ } -> record_derived_path path
        | _ -> ());
        super#signature_item item

      (* A deriving payload names the derivers ([equal], [compare], [hash]), which for a type [t]
         are exactly its derived names; it uses nothing. *)
      method! attribute attribute =
        if String.equal attribute.attr_name.txt "deriving" then () else super#attribute attribute
    end
  in
  List.iter implementations ~f:(fun (source, contents) ->
      (walk ~source)#structure (Read.structure_of contents));
  List.iter interfaces ~f:(fun (source, contents) ->
      (walk ~source)#signature (Ppxlib.Parse.interface (Lexing.from_string contents)));
  let table = Hashtbl.create (module String) in
  List.iter type_exports ~f:(fun export ->
      let start, stop = export.span in
      let outside_declaration (source, position) =
        not (String.equal source export.source && start <= position && position < stop)
      in
      let by_type =
        Hashtbl.find_multi type_mentions export.type_name |> List.count ~f:outside_declaration
      in
      let by_derived =
        List.sum
          (module Int)
          export.mentioned_by
          ~f:(fun name ->
            Hashtbl.find_multi derived_mentions name |> List.count ~f:outside_declaration)
      in
      Hashtbl.update table (type_export_key export) ~f:(fun previous ->
          Option.value previous ~default:0 + by_type + by_derived));
  table
