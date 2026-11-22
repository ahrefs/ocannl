(* The eager keys in Utils are the literal ~arg_name reads in assignments to settings fields
   (Config_key_scan.settings_field_initializers) and in the top-level c_stderr_detached binding.
   Every field of the top-level settings record must have a recognized read. The bootstrap keys
   already have their own list in Utils. This rule does not follow calls out of those expressions or
   read lazy settings, other bindings, or other modules. *)

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

let detached_stderr_keys source structure =
  let bindings =
    List.concat_map structure ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.filter bindings ~f:(fun binding ->
                Option.equal String.equal
                  (Config_key_scan.pattern_name binding.pvb_pat)
                  (Some "c_stderr_detached"))
        | _ -> [])
  in
  let keys =
    match bindings with
    | [ binding ] ->
        let loc = binding.pvb_expr.pexp_loc in
        Config_key_scan.keyed_reads source
        |> List.filter_map ~f:(fun read ->
            Option.some_if
              (loc.loc_start.pos_cnum <= read.Config_key_scan.at && read.at < loc.loc_end.pos_cnum)
              read.read_key)
    | _ -> []
  in
  if List.is_empty keys then failwith "no eager key read by Utils.c_stderr_detached";
  keys

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
    @ detached_stderr_keys source structure
  in
  Stdio.printf "let keys = [\n";
  List.iter (List.dedup_and_sort keys ~compare:String.compare) ~f:(Stdio.printf "  %S;\n");
  Stdio.printf "]\n"
