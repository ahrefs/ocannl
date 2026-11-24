(* The census reads the parsed Utils implementation. It credits literal ~arg_name arguments of the
   getters below in top-level value initializers or evaluation items, excluding function, lazy,
   object, module and extension bodies. A getter site outside that walk keeps its key out of the
   startup-only set. It does not follow calls or aliases, expand PPX, infer what a callback does, or
   inspect other files. The settings-field census separately covers restore_settings and requires a
   read for every field. Bootstrap keys are already owned by profile_ineligible_keys in Utils. *)

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

let startup_only_keys source structure =
  let eager_offsets = Hash_set.create (module Int) in
  let walk =
    object
      inherit Ppxlib.Ast_traverse.iter as super
      method! attribute _ = ()
      method! module_expr _ = ()

      method! expression expr =
        match expr.pexp_desc with
        | Pexp_function _ | Pexp_lazy _ | Pexp_object _ | Pexp_extension _ -> ()
        | _ ->
            Hash_set.add eager_offsets expr.pexp_loc.loc_start.pos_cnum;
            super#expression expr
    end
  in
  List.iter structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) -> List.iter bindings ~f:(fun b -> walk#expression b.pvb_expr)
      | Pstr_eval (expr, _) -> walk#expression expr
      | _ -> ());
  let reads =
    Config_key_scan.keyed_reads source
    |> List.filter ~f:(fun read ->
        match read.Config_key_scan.reader with
        | Some [ ("get_global_arg" | "get_global_arg_with_source" | "get_global_flag") ] ->
            read.defaulted
        | Some
            [
              ( "registered_flag" | "registered_int" | "registered_float" | "registered_keyword"
              | "registered_text" | "registered_positive_float_or_unset"
              | "registered_float_at_least" );
            ] ->
            true
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
