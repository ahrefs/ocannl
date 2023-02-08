(* This file is inspired by:
   https://github.com/metaocaml/ber-metaocaml/blob/ber-n111/ber-metaocaml-111/runnative.ml *)

open Base

let load_path : string list ref = ref []

(** Add a directory to search for .cmo/.cmi files, needed for the sake of running the generated code.
    The specified directory is prepended to [load_path]. *)
let add_search_path dir =
  load_path := dir :: !load_path

let ocamlopt_path = "ocamlfind ocamlopt"

(** Compile the source file and make the .cmxs, returning its name. *)
let compile_source ~with_debug src_fname =
  let basename = Caml.Filename.remove_extension src_fname in
  let plugin_fname = basename ^ ".cmxs" in
  let other_files = [basename ^ ".cmi"; basename ^ ".cmx"(*; basename ^ Config.ext_obj*)] in
  (* We need the byte objects directory in path because it contains the .cmi files. *)
  (* FIXME: un-hardcode the paths. *)
  let cmdline = ocamlopt_path ^ 
                " -I ~/ocannl/_build/default/lib -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/native -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/byte -package base -package stdio " ^
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

(** Create a file to compile and later link, using the given closed code. *)
let create_comp_unit closed =
  let fname, oc =
    Caml.Filename.open_temp_file ~mode:[Open_wronly;Open_creat;Open_text]
      code_file_prefix ".ml" in
  (* FIXME(32): the following outputs truncated source code -- missing the last line:
  let ppf = Caml.Format.formatter_of_out_channel oc in
  Caml.Format.pp_set_margin ppf 160;
  let () = EmitOCaml.format_code ppf closed in
  let () = Stdio.Out_channel.close oc in *)
  Caml.Format.pp_set_margin Caml.Format.str_formatter 160;
  EmitOCaml.format_code Caml.Format.str_formatter closed;
  let contents = Caml.Format.flush_str_formatter() in
  Stdio.Out_channel.output_string oc contents;
  Stdio.Out_channel.flush oc;
  Stdio.Out_channel.close oc;
  (* Stdio.printf "\nCreated file:\n%s\n\n%!" @@ Stdio.In_channel.read_all fname; *)
  fname

let first_file_span_re = Str.regexp @@
  code_file_prefix ^ "[A-Za-z0-9]*.ml\", line \\([0-9]+\\), characters \\([0-9]+\\)-\\([0-9]+\\)"

(** Returns the character offset span inside [contents] corresponding to the first file span from [message].
    Returns [0, 0] if no span is found. *)
let first_file_span ~contents ~message =
  let last_char = String.length contents - 1 in
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

let error_opening_delimiter = " {$# "
let error_closing_delimiter = " #$} "

let handle_error prefix ?formula ~contents exc =
  let message = Caml.Printexc.to_string exc^"\n"^Caml.Printexc.get_backtrace() in
  let from_pos, to_pos = first_file_span ~contents ~message in
  let contents =
    String.sub contents ~pos:0 ~len:from_pos ^ error_opening_delimiter ^
    String.sub contents ~pos:from_pos ~len:(to_pos - from_pos) ^ error_closing_delimiter ^ 
    String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos) in
  let contents = String.substr_replace_all contents ~pattern:"Ocannl_runtime." ~with_:"" in
  let contents = String.substr_replace_all contents ~pattern:"Node." ~with_:"" in
  let contents = String.substr_replace_all contents ~pattern:"Base." ~with_:"" in
  let contents = String.substr_replace_all contents ~pattern:"Stdlib.Bigarray.Genarray." ~with_:"A." in
  let exc = Formula.Session_error (
    prefix^"\n"^message^"\nIn code span "^error_opening_delimiter^"..."^error_closing_delimiter^
    ":\n"^contents, formula) in
  Stdio.prerr_endline @@ Option.value_exn (Formula.session_error_printer exc);
  raise exc

let load_native ?(with_debug=true) (cde: Code.program) =
  let closed = EmitOCaml.emit cde in
  if not Dynlink.is_native then invalid_arg "Exec.load_forward: only works in native code";
  let source_fname = create_comp_unit closed in
  let plugin_fname = compile_source ~with_debug source_fname in
  let result =
    if with_debug then (
      Caml.Format.pp_set_margin Caml.Format.str_formatter 160;
      EmitOCaml.format_code Caml.Format.str_formatter closed;
      let contents = Caml.Format.flush_str_formatter() in
      try Dynlink.loadfile_private plugin_fname; Some contents with
      | Dynlink.Error (Library's_module_initializers_failed exc) ->
        handle_error "Runtime init error:" ~contents exc)

    else (Dynlink.loadfile_private plugin_fname; None) in
  Caml.Sys.remove plugin_fname;
  Caml.Sys.remove source_fname;
  result
