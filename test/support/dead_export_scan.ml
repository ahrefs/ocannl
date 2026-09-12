(** The source reader behind the dead implicit-export ratchet.

    This is deliberately a first cut. An implementation without an [.mli] exports every
    source-declared top-level [let] and [external], including helpers intended only for the
    implementation. Source-declared [let] values whose name starts with [_] are excluded: the author
    already marked them deliberately unused, matching OCaml warning 32. This also covers patterns
    and extension payloads; externals and inferred deriving names retain their existing census
    policy, including polymorphic-variant parser helpers. We enumerate those declarations in
    [arrayjit/lib/] and [tensor/], then count references from every other OCaml source.

    A reference is conservative: a direct qualified path ([M.v]), a path through a module alias, or
    an unqualified identifier inside the lexical range of [open M]. Alias scopes are deliberately
    over-approximated to the whole source, and an [include M] counts as a reference to every value
    because it re-exports the whole interface. Both choices can hide a dead export through a false
    positive, but cannot falsely reject an ordinary use. Values generated for top-level types by
    [of_sexp], [compare], and [equal] derivings are included (a standalone [of_sexp] deriving as
    much as the [of_sexp] half of [sexp]); their expression extensions count as references without
    needing to spell the generated value. The [sexp_of] converters -- derived or hand-written,
    recognized by the [sexp_of_] name prefix -- are excluded by policy: they are the entry point for
    debugging and observability, and consumed as often through [ppx_minidebug]'s typed log
    annotations, which expand to converter calls this source-level census cannot see, as through
    spelled references. A [sexp_of] with no caller costs nothing and cannot drift from its type,
    while removing it to satisfy a ratchet only takes the converter away from the next debugging
    session. Values introduced by other PPX expansions or by an [include] inside the defining module
    remain outside this source-level census. *)

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

(** References to [exports] from [sources]. Sources are [(repository-relative path, contents)]. *)
let references ~(exports : export list) ~sources =
  let modules =
    List.map exports ~f:(fun export -> export.module_name)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let export_names =
    List.fold exports
      ~init:(Map.empty (module String))
      ~f:(fun map export ->
        Map.update map export.module_name ~f:(function
          | None -> Set.singleton (module String) export.value
          | Some values -> Set.add values export.value))
  in
  List.concat_map sources ~f:(fun (source, contents) ->
      let structure = Read.structure_of contents in
      List.concat_map modules ~f:(fun module_name ->
          let aliases, _opened, open_ranges =
            Read.module_bindings_of structure ~wanted:module_name
          in
          let receivers = Set.of_list (module String) (module_name :: aliases) in
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
                (* [include M] consumes and re-exports M's entire interface. Count every value; an
                   external caller may subsequently reach it only through the including module. *)
                (match item.pstr_desc with
                | Pstr_include { pincl_mod; _ } -> (
                    match module_expr_name pincl_mod with
                    | Some receiver when Set.mem receivers receiver ->
                        Set.iter names ~f:(fun value ->
                            add ~value ~line:item.pstr_loc.loc_start.pos_lnum
                              ~spelling:("include " ^ receiver))
                    | Some _ | None -> ())
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
