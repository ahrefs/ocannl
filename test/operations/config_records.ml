(* gh-ocannl-1566: the resolved-configuration records, as [Utils.resolved_config] returns them and
   as [Utils.config_record_to_json] renders them for the [log_config_json] stream.

   The rule runs this with [--ocannl_autotune_timing=isolated] and [--ocannl_profile=approximate] on
   its commandline and this directory's [ocannl_config] beside it, so every source kind but the
   environment is reached by a real resolution; the environment one is reached by setting a variable
   mid-run, which is also the changed-resolution case. That the stream carries these same records is
   checked where a stream exists, in the bin_cells control (test/operations/bin_cells.py, the
   probe's [records] mode). The precedence walk itself is config_profiles' business: a record
   carries [resolve_config_value]'s source unchanged. *)

open Base
open Stdio
open Verdict.Claims

let records_of key =
  List.filter (Utils.resolved_config ()) ~f:(fun r -> String.equal r.Utils.key key)

let json = Utils.config_record_to_json

let () =
  (* The bootstrap settings come first, from the resolutions the bootstrap made. *)
  let bootstrap =
    [
      "log_config_sourcing";
      "log_config_json";
      "no_config_file";
      "suppress_welcome_message";
      "profile";
    ]
  in
  p "the bootstrap settings are the first records, in their resolution order"
    (List.equal String.equal bootstrap
       (List.map (List.take (Utils.resolved_config ()) (List.length bootstrap)) ~f:(fun r -> r.key)));
  p "the profile picked on the commandline is recorded with its argv element"
    (match records_of "profile" with
    | [ { value = "approximate"; source = From_cmdline "--ocannl_profile=approximate"; _ } ] -> true
    | _ -> false);
  (* A commandline setting, read twice. *)
  let timing () = Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued" in
  let first = timing () in
  let second = timing () in
  p "a commandline setting is recorded once, with the argv element that stated it"
    (String.equal first second
    &&
    match records_of "autotune_timing" with
    | [ { value = "isolated"; source = From_cmdline "--ocannl_autotune_timing=isolated"; _ } ] ->
        true
    | _ -> false);
  p "a commandline record renders its argv element as the origin, and its source as its layer"
    (List.equal String.equal
       (List.map (records_of "autotune_timing") ~f:json)
       [
         {|{"ocannl_config":1,"key":"autotune_timing","value":"isolated","source":"commandline","layer":"commandline","profile":null,"origin":"--ocannl_autotune_timing=isolated"}|};
       ]);
  (* A profile-payload value, through a typed getter: the raw string is recorded. *)
  let rounds = Utils.registered_int ~arg_name:"autotune_rounds" in
  p "a typed getter's read is recorded, the profile value under the level that picked it"
    (rounds = 4
    &&
    match records_of "autotune_rounds" with
    | [ ({ value = "4"; source = From_profile (Cmdline_level, "approximate"); _ } as r) ] ->
        String.equal (json r)
          {|{"ocannl_config":1,"key":"autotune_rounds","value":"4","source":"profile","layer":"commandline","profile":"approximate","origin":"--ocannl_profile=approximate"}|}
    | _ -> false);
  (* The config file: this directory's, which states suppress_welcome_message among others. *)
  let file_records =
    List.filter (Utils.resolved_config ()) ~f:(fun r ->
        match r.source with From_config_file -> true | _ -> false)
  in
  p_all "every config-file record names, as its origin, a file stating exactly that key=value"
    file_records ~f:(fun r ->
      let path = Utils.config_file_path in
      String.equal (Stdlib.Filename.basename path) "ocannl_config"
      && List.exists (In_channel.read_lines path) ~f:(fun line ->
          String.equal (String.strip line) (r.key ^ "=" ^ r.value))
      && String.is_suffix (json r) ~suffix:(",\"origin\":" ^ Utils.json_string_literal path ^ "}"));
  (* A setting nobody states, then the environment changing under the process. *)
  let pcre () = Utils.get_global_arg ~arg_name:"debug_highlight_pcre" ~default:"" in
  let before = pcre () in
  Unix.putenv "OCANNL_DEBUG_HIGHLIGHT_PCRE" "zz";
  let after = pcre () in
  let _again : string = pcre () in
  p "a default read and then a changed resolution are two records, the change last"
    (String.is_empty before && String.equal after "zz"
    &&
    match records_of "debug_highlight_pcre" with
    | [
     ({ value = ""; source = From_default; _ } as d);
     ({ value = "zz"; source = From_env "OCANNL_DEBUG_HIGHLIGHT_PCRE"; _ } as e);
    ] ->
        String.is_suffix (json d)
          ~suffix:{|"source":"default","layer":"default","profile":null,"origin":null}|}
        && String.is_suffix (json e)
             ~suffix:
               {|"source":"environment","layer":"environment","profile":null,"origin":"OCANNL_DEBUG_HIGHLIGHT_PCRE"}|}
    | _ -> false);
  p "reading the records records nothing"
    (List.length (Utils.resolved_config ()) = List.length (Utils.resolved_config ()))

(* The encoder on values the old prose trace could not carry unambiguously. *)
let () =
  printf "Rendered records:\n";
  let samples =
    [
      {
        Utils.key = "debug_highlight_pcre";
        value = "a\"b\\c\td\ne\001f";
        source = From_cmdline "--ocannl_debug_highlight_pcre=a\"b";
      };
      {
        key = "debug_highlight_pcre";
        value = "x, commandline y";
        source = From_cmdline "--ocannl_debug_highlight_pcre=x, commandline y";
      };
      {
        key = "debug_highlight_pcre";
        value = "za\xc5\xbc\xc3\xb3\xc5\x82\xc4\x87";
        source = From_env "OCANNL_DEBUG_HIGHLIGHT_PCRE";
      };
      { key = "debug_highlight_pcre"; value = ""; source = From_default };
      {
        key = "debug_highlight_pcre";
        value = "true";
        source = From_profile (Cmdline_level, "approximate");
      };
    ]
  in
  let lines = List.map samples ~f:json in
  List.iter lines ~f:print_endline;
  p_all "every rendered record is one line, free of control characters" lines ~f:(fun l ->
      not (String.exists l ~f:(fun c -> Char.to_int c < 0x20)))
