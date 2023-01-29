(* This file is inspired by:
   https://github.com/metaocaml/ber-metaocaml/blob/ber-n111/ber-metaocaml-111/runnative.ml *)

open Base

let load_path : string list ref = ref []

(* Add a directory to search for .cmo/.cmi files, needed
   for the sake of running the generated code .
   The specified directory is prepended to the load_path.
*)
let add_search_path dir =
  load_path := dir :: !load_path

let ocamlopt_path = 
  let open Caml.Filename in
  concat (concat (dirname (dirname (Config.standard_library))) "bin") "ocamlfind ocamlopt"

(* Compile the source file and make the .cmxs, returning its name *)
let compile_source ~with_debug src_fname =
  let basename = Caml.Filename.remove_extension src_fname in
  let plugin_fname =  basename ^ ".cmxs" in
  let other_files  =  [basename ^ ".cmi"; basename ^ ".cmx";
                       basename ^ Config.ext_obj] in
  (* We need the byte objects directory in path because it contains the .cmi files. *)
  (* FIXME: un-hardcode the paths. *)
  let cmdline = ocamlopt_path ^ 
                " -I ~/ocannl/_build/default/lib -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/native -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/byte -package base " ^
                " -shared"^(if with_debug then " -g" else "")^" -o " ^ plugin_fname ^
                (String.concat ~sep:"" @@ 
                 List.map ~f:(fun p -> " -I " ^ p) !load_path) ^
                " " ^ src_fname in      
  let rc = Caml.Sys.command cmdline in
  List.iter ~f:Caml.Sys.remove other_files;
  if rc = 0 then plugin_fname else 
    let () = Caml.Sys.remove plugin_fname in
    failwith "Ocannl.exec: .cmxs compilation failure"

let code_file_prefix = "runn"

(* Create a file to compile and later link, using the given closed code *)
let create_comp_unit ~with_debug closed =
  let fname, oc =
    Caml.Filename.open_temp_file ~mode:[Open_wronly;Open_creat;Open_text]
      code_file_prefix ".ml" in
  let ppf = Caml.Format.formatter_of_out_channel oc in
  Caml.Format.pp_set_margin ppf 160;
  let () =
  (* TODO(29): Cleanup when the backtrace can be captured in this file. *)
    if with_debug then
      Caml.Format.fprintf ppf
       "try@ %a@ with error -> Ocannl_runtime.Node.set_error_message error; raise error@.%!"
       Codelib.format_code closed
    else Codelib.format_code ppf closed in
  let () = Stdio.Out_channel.close oc in
  fname

let first_file_span_re = Str.regexp "line \\([0-9]+\\), characters \\([0-9]+\\)-\\([0-9]+\\)"

(** Returns the character offset span inside [contents] corresponding to the first file span from [message].
    Returns [0, 0] if no span is found. *)
let first_file_span ~contents ~message =
  let last_char = String.length contents - 1 in
  (* Stdio.printf "first_file_span: message=%s, match=%d\n" message (Str.search_forward first_file_span_re message 0); *)
  try
    ignore (Str.search_forward first_file_span_re message 0);
    let line_num = Int.of_string @@ Str.matched_group 1 message in
    let char_start = Int.of_string @@ Str.matched_group 2 message in
    let char_end = Int.of_string @@ Str.matched_group 3 message in
    let rec line_offset ~line_num ~from =
      if line_num <= 1 then from else
        match String.index_from contents from '\n' with
        | None -> from
        | Some pos -> line_offset ~line_num:(line_num-1) ~from:(pos+1) in
    let line_offset = line_offset ~line_num ~from:0 in
    line_offset + char_start, line_offset + char_end
  with Caml.Not_found ->
    0, last_char

let load_native ?(with_debug=true) (cde: unit Codelib.code) =
  let closed = Codelib.close_code cde in
  if not Dynlink.is_native then invalid_arg "Exec.load_forward: only works in native code";
  let source_fname = create_comp_unit ~with_debug closed in
  let plugin_fname = compile_source ~with_debug source_fname in
  let () =
    if with_debug then
      try Dynlink.loadfile_private plugin_fname with
      | Dynlink.Error (Library's_module_initializers_failed _exc) ->
        Caml.Format.pp_set_margin Caml.Format.str_formatter 160;
        Caml.Format.fprintf Caml.Format.str_formatter
          "try@ %a@ with error -> Ocannl_runtime.Node.set_error_message error; raise error@.%!"
          Codelib.format_code closed;
        let contents = Caml.Format.flush_str_formatter() in
        let message =
          match !Ocannl_runtime.Node.error_message__ with None -> "error not captured" | Some m -> m in
        Ocannl_runtime.Node.error_message__ := None;
        let from_pos, to_pos = first_file_span ~contents ~message in
        let contents =
          String.sub contents ~pos:0 ~len:from_pos ^ "«" ^
          String.sub contents ~pos:from_pos ~len:(to_pos - from_pos) ^ "»" ^ 
          String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos) in
        let contents = String.substr_replace_all contents ~pattern:"Ocannl_runtime." ~with_:"" in
        let contents = String.substr_replace_all contents ~pattern:"Node." ~with_:"" in
        raise @@ Formula.Session_error ("Runtime error: \n"^message^"\nIn code span «...»:\n"^contents, None)
    
    else Dynlink.loadfile_private plugin_fname in
  Caml.Sys.remove plugin_fname;
  Caml.Sys.remove source_fname
