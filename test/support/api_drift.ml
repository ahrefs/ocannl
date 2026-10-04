open Base
(** Source declarations for editorial review. Implementations retain their bodies: without type
    checking, a body change may change an inferred public type. PPX-generated exports and inferred
    types must still be checked by the editor; this reader does not claim ABI coverage. *)

open Ppxlib.Parsetree

type declaration = { name : string; line : int; text : string }

let binding_names pattern =
  let modules = ref [] in
  let iterator =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with
        | Ppat_unpack { txt = Some name; _ } -> modules := name :: !modules
        | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  Dead_export_scan.pattern_names pattern @ !modules

let derived_inputs ~paths dunes =
  let present = Set.of_list (module String) paths in
  List.concat_map dunes ~f:(fun (dune_path, contents) ->
      let directory = Stdlib.Filename.dirname dune_path in
      let path name = directory ^ "/" ^ name in
      let stanzas = Dune_stanza_scan.stanzas contents in
      let rec select_inputs = function
        | Sexp.List (Sexp.Atom "select" :: Sexp.Atom output :: Sexp.Atom "from" :: clauses) ->
            List.map clauses ~f:(function
              | Sexp.List terms -> (
                  match List.rev terms with
                  | Sexp.Atom input :: Sexp.Atom "->" :: _ -> (path input, path output)
                  | _ -> failwith ("unsupported select clause in " ^ dune_path))
              | _ -> failwith ("unsupported select clause in " ^ dune_path))
        | Sexp.List children -> List.concat_map children ~f:select_inputs
        | Sexp.Atom _ -> []
      in
      let generators =
        List.concat_map stanzas ~f:(fun stanza ->
            let spec =
              match Dune_stanza_scan.head stanza with
              | Some "ocamllex" ->
                  let modules =
                    match Dune_stanza_scan.field stanza "modules" with
                    | Some modules -> modules
                    | None -> ( match stanza with Sexp.List (_ :: modules) -> modules | _ -> [])
                  in
                  Some (".mll", modules, None)
              | Some "menhir" ->
                  let modules =
                    Option.value (Dune_stanza_scan.field stanza "modules") ~default:[]
                  in
                  let output =
                    match Dune_stanza_scan.field stanza "merge_into" with
                    | Some [ Sexp.Atom name ] -> Some name
                    | None -> None
                    | Some _ -> failwith ("unsupported menhir merge_into in " ^ dune_path)
                  in
                  Some (".mly", modules, output)
              | _ -> None
            in
            match spec with
            | None -> []
            | Some (suffix, modules, output) ->
                List.map modules ~f:(function
                  | Sexp.Atom name
                    when Option.is_some (Dead_export_scan.module_name_of_source (name ^ ".ml")) ->
                      (path (name ^ suffix), path (Option.value output ~default:name ^ ".ml"))
                  | _ -> failwith ("unsupported generator module set in " ^ dune_path)))
        @ List.concat_map stanzas ~f:select_inputs
      in
      let directory_modules =
        List.filter_map paths ~f:(fun p ->
            if String.equal (Stdlib.Filename.dirname p) directory then
              Dead_export_scan.module_name_of_source p
            else None)
        @ List.map generators ~f:(fun (_, output) ->
            Option.value_exn (Dead_export_scan.module_name_of_source output))
        |> List.map ~f:String.lowercase
        |> List.dedup_and_sort ~compare:String.compare
      in
      let owners =
        List.filter stanzas ~f:(fun s ->
            List.mem
              [ "library"; "executable"; "executables"; "test"; "tests" ]
              (Option.value (Dune_stanza_scan.head s) ~default:"")
              ~equal:String.equal)
      in
      let public_modules =
        List.filter owners ~f:(fun s ->
            Option.equal String.equal (Dune_stanza_scan.head s) (Some "library")
            && not (List.is_empty (Dune_stanza_scan.public_names s)))
        |> List.concat_map ~f:(Dune_stanza_scan.modules_of ~directory_modules owners)
        |> List.map ~f:String.lowercase
        |> Set.of_list (module String)
      in
      List.filter_map generators ~f:(fun (input, output) ->
          let name =
            Option.value_exn (Dead_export_scan.module_name_of_source output) |> String.lowercase
          in
          let interface = String.chop_suffix_exn output ~suffix:".ml" ^ ".mli" in
          Option.some_if
            (Set.mem present input && Set.mem public_modules name && not (Set.mem present interface))
            input))

let sources ?(dunes = []) paths =
  List.filter paths ~f:(fun path ->
      Dead_export_scan.in_scan_root path && String.is_suffix path ~suffix:".mli")
  @ List.filter (Dead_export_scan.implicit_implementations paths) ~f:(fun source ->
      Option.is_some (Dead_export_scan.module_name_of_source source))
  @ derived_inputs ~paths dunes
  |> List.dedup_and_sort ~compare:String.compare

let declarations ~source contents =
  if String.is_suffix source ~suffix:".mll" || String.is_suffix source ~suffix:".mly" then
    (* This is deliberately a review entry for a generator INPUT. Generating and typechecking a
       historical module would require its historical dependency tree and toolchain. Keep all input
       changes visible, without claiming to reconstruct its generated/inferred interface. *)
    [ { name = "generated module input[0]"; line = 1; text = contents } ]
  else
    let counts = Hashtbl.create (module String) in
    let entry name loc text =
      let ordinal = Hashtbl.find_or_add counts name ~default:(fun () -> 0) in
      Hashtbl.set counts ~key:name ~data:(ordinal + 1);
      {
        name = Printf.sprintf "%s[%d]" name ordinal;
        line = loc.Ppxlib.Location.loc_start.pos_lnum;
        text;
      }
    in
    let type_names declarations =
      "type " ^ String.concat ~sep:"," (List.map declarations ~f:(fun t -> t.ptype_name.txt))
    in
    let signature_item item =
      let name =
        match item.psig_desc with
        | Psig_value v -> "val " ^ v.pval_name.txt
        | Psig_type (_, ts) | Psig_typesubst ts -> type_names ts
        | Psig_typext t -> "type extension " ^ Ppxlib.Longident.name t.ptyext_path.txt
        | Psig_exception e -> "exception " ^ e.ptyexn_constructor.pext_name.txt
        | Psig_module m -> "module " ^ Option.value m.pmd_name.txt ~default:"_"
        | Psig_recmodule ms ->
            "module rec " ^ String.concat ~sep:"," (List.filter_map ms ~f:(fun m -> m.pmd_name.txt))
        | Psig_modtype m | Psig_modtypesubst m -> "module type " ^ m.pmtd_name.txt
        | Psig_modsubst m -> "module substitution " ^ m.pms_name.txt
        | Psig_include _ -> "include"
        | Psig_class cs ->
            "class " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Psig_class_type cs ->
            "class type " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Psig_extension ((name, _), _) -> "extension " ^ name.txt
        | Psig_open _ | Psig_attribute _ -> "environment"
      in
      entry name item.psig_loc (Stdlib.Format.asprintf "%a" Ppxlib.Pprintast.signature [ item ])
    in
    let structure_item item =
      let name =
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            "let "
            ^ String.concat ~sep:","
                (List.concat_map bindings ~f:(fun b -> binding_names b.pvb_pat))
        | Pstr_primitive v -> "external " ^ v.pval_name.txt
        | Pstr_type (_, ts) -> type_names ts
        | Pstr_typext t -> "type extension " ^ Ppxlib.Longident.name t.ptyext_path.txt
        | Pstr_exception e -> "exception " ^ e.ptyexn_constructor.pext_name.txt
        | Pstr_module m -> "module " ^ Option.value m.pmb_name.txt ~default:"_"
        | Pstr_recmodule ms ->
            "module rec " ^ String.concat ~sep:"," (List.filter_map ms ~f:(fun m -> m.pmb_name.txt))
        | Pstr_modtype m -> "module type " ^ m.pmtd_name.txt
        | Pstr_include _ -> "include"
        | Pstr_class cs ->
            "class " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Pstr_class_type cs ->
            "class type " ^ String.concat ~sep:"," (List.map cs ~f:(fun c -> c.pci_name.txt))
        | Pstr_extension ((name, _), _) -> "extension " ^ name.txt
        | Pstr_open _ | Pstr_attribute _ -> "environment"
        | Pstr_eval _ -> "evaluation"
      in
      entry name item.pstr_loc (Stdlib.Format.asprintf "%a" Ppxlib.Pprintast.structure [ item ])
    in
    let lexbuf = Lexing.from_string contents in
    Ppxlib.Location.init lexbuf source;
    (* Keep attributes that can generate or alter public surface, but discard prose. *)
    let strip_docs =
      object
        inherit Ppxlib.Ast_traverse.map as super
        val mutable prune_nonexports = true

        (* An extension may turn any payload into exported declarations. Keep its input visible;
           dropping a payload evaluation would silently turn [%%publish earlier] and [%%publish
           later] into the same empty extension. *)
        method! extension extension =
          let saved = prune_nonexports in
          prune_nonexports <- false;
          Exn.protect
            ~f:(fun () -> super#extension extension)
            ~finally:(fun () -> prune_nonexports <- saved)

        method! attributes attrs =
          super#attributes
            (List.filter attrs ~f:(fun a ->
                 not (List.mem [ "ocaml.doc"; "ocaml.text" ] a.attr_name.txt ~equal:String.equal)))

        method! signature items =
          super#signature
            (List.filter items ~f:(fun item ->
                 match item.psig_desc with
                 | Psig_attribute a ->
                     not
                       (List.mem [ "ocaml.doc"; "ocaml.text" ] a.attr_name.txt ~equal:String.equal)
                 | _ -> true))

        method! structure items =
          super#structure
            (List.filter items ~f:(fun item ->
                 match item.pstr_desc with
                 | Pstr_eval _ when prune_nonexports -> false
                 | Pstr_value (_, bindings) when prune_nonexports ->
                     List.exists bindings ~f:(fun binding ->
                         not (List.is_empty (binding_names binding.pvb_pat)))
                 | Pstr_attribute a ->
                     not
                       (List.mem [ "ocaml.doc"; "ocaml.text" ] a.attr_name.txt ~equal:String.equal)
                 | _ -> true))
      end
    in
    if String.is_suffix source ~suffix:".mli" then
      Ppxlib.Parse.interface lexbuf |> strip_docs#signature |> List.map ~f:signature_item
    else Ppxlib.Parse.implementation lexbuf |> strip_docs#structure |> List.map ~f:structure_item

let changes before after =
  let map declarations =
    Map.of_alist_exn (module String) (List.map declarations ~f:(fun d -> (d.name, d)))
  in
  Map.merge (map before) (map after) ~f:(fun ~key:_ -> function
    | `Both (a, b) when String.equal a.text b.text -> None
    | `Both (a, b) -> Some (Some a, Some b)
    | `Left a -> Some (Some a, None)
    | `Right b -> Some (None, Some b))
  |> Map.data
