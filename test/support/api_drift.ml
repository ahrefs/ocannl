open Base
(** Source declarations for editorial review. Implementations retain their bodies: without type
    checking, a body change may change an inferred public type. PPX-generated exports and inferred
    types must still be checked by the editor; this reader does not claim ABI coverage. *)

open Ppxlib.Parsetree

type declaration = { name : string; line : int; text : string }

let sources paths =
  List.filter paths ~f:(fun path ->
      Dead_export_scan.in_scan_root path && String.is_suffix path ~suffix:".mli")
  @ Dead_export_scan.implicit_implementations paths
  |> List.dedup_and_sort ~compare:String.compare

let declarations ~source contents =
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
              (List.concat_map bindings ~f:(fun b -> Dead_export_scan.pattern_names b.pvb_pat))
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

      method! attributes attrs =
        super#attributes
          (List.filter attrs ~f:(fun a ->
               not (List.mem [ "ocaml.doc"; "ocaml.text" ] a.attr_name.txt ~equal:String.equal)))
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
