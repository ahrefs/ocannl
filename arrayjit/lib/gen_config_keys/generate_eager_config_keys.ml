(* The census reads literal ~arg_name arguments of unqualified, fully applied getters in the parsed
   Utils implementation. The eager walk starts with value initializers and evaluation items. It
   enters ordinary module structures through module bindings, local modules, opens, includes and
   packs. The shared parser inlines a locally defined functor included exactly once at the top
   level. The walk stops at other module applications and at signatures, function, functor, lazy,
   object, attribute and extension bodies. A getter site outside the walk keeps its key out of the
   startup-only set. The census operates on direct calls in this file, without following aliases or
   callbacks or expanding PPX. Typed getter names come from the literal arms of
   registered_getter_of_kind. The settings-field census separately covers restore_settings and
   requires a read for every field. Bootstrap keys are owned by profile_ineligible_keys in Utils. *)

open Base
open Ppxlib.Parsetree

let settings_fields structure =
  let declarations =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_type (_, declarations) ->
            List.filter declarations ~f:(fun decl -> String.equal decl.ptype_name.txt "settings")
        | _ -> [])
  in
  match declarations with
  | [ { ptype_kind = Ptype_record fields; _ } ] ->
      List.map fields ~f:(fun field -> field.pld_name.txt)
  | _ -> failwith "expected a single top-level Utils.settings record type"

let registered_getters structure =
  let registry =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.filter bindings ~f:(fun binding ->
                match Config_key_scan.pattern_name binding.pvb_pat with
                | Some "registered_getter_of_kind" -> true
                | _ -> false)
        | _ -> [])
  in
  match registry with
  | [ { pvb_expr = { pexp_desc = Pexp_function ([], _, Pfunction_cases (cases, _, _)); _ }; _ } ] ->
      List.map cases ~f:(fun case ->
          match Config_key_scan.string_literal case.pc_rhs with
          | Some getter -> getter
          | None -> failwith "expected a literal getter name in Utils.registered_getter_of_kind")
      |> Set.of_list (module String)
  | _ -> failwith "expected a single top-level Utils.registered_getter_of_kind function"

let startup_only_keys source structure =
  let registered_getters = registered_getters structure in
  let eager_offsets = Hash_set.create (module Int) in
  let walk =
    object
      inherit Ppxlib.Ast_traverse.iter as super
      method! attribute _ = ()
      method! module_type _ = ()

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_value _ | Pstr_eval _ | Pstr_module _ | Pstr_recmodule _ | Pstr_open _
        | Pstr_include _ ->
            super#structure_item item
        | _ -> ()

      method! module_expr expr =
        match expr.pmod_desc with
        | Pmod_structure _ | Pmod_constraint _ -> super#module_expr expr
        | _ -> ()

      method! expression expr =
        match expr.pexp_desc with
        | Pexp_function _ | Pexp_lazy _ | Pexp_object _ | Pexp_extension _ -> ()
        | _ ->
            Hash_set.add eager_offsets expr.pexp_loc.loc_start.pos_cnum;
            super#expression expr
    end
  in
  walk#structure structure;
  let reads =
    Config_key_scan.keyed_reads source
    |> List.filter ~f:(fun read ->
        match read.Config_key_scan.reader with
        | Some [ ("get_global_arg" | "get_global_arg_with_source" | "get_global_flag") ] ->
            read.defaulted
        | Some [ getter ] -> (not read.defaulted) && Set.mem registered_getters getter
        | _ -> false)
  in
  let eager, other = List.partition_tf reads ~f:(fun read -> Hash_set.mem eager_offsets read.at) in
  let keys reads =
    List.map reads ~f:(fun read -> read.Config_key_scan.read_key) |> Set.of_list (module String)
  in
  Set.to_list (Set.diff (keys eager) (keys other))

let () =
  let source = Stdio.In_channel.read_all Stdlib.Sys.argv.(1) in
  let structure = Config_key_scan.structure_of source in
  let initializers =
    Config_key_scan.settings_field_initializers source |> Map.of_alist_multi (module String)
  in
  let keys =
    List.concat_map (settings_fields structure) ~f:(fun field ->
        match List.concat (Map.find_multi initializers field) with
        | [] -> failwith ("no eager key initializes Utils.settings." ^ field)
        | keys -> keys)
    @ startup_only_keys source structure
  in
  Stdio.printf "let keys = [\n";
  List.iter (List.dedup_and_sort keys ~compare:String.compare) ~f:(Stdio.printf "  %S;\n");
  Stdio.printf "]\n"
