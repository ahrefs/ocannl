open Base

let emit = Code.unoptimized_program

let semi ppf () = Caml.Format.fprintf ppf ";@ "

let pp_print_init_op ppf: Code.init_op -> unit = function
  | `Unspecified -> Caml.Format.pp_print_string ppf "`Unspecified"
  | `ConstantOfValue c ->
    Caml.Format.fprintf ppf "(`ConstantOfValue %f)" c
  | `FixedConstant cs ->
    Caml.Format.(fprintf ppf "(`FixedConstant @[<2>[|%a|]@])"
                   (pp_print_list ~pp_sep:semi pp_print_float) @@ Array.to_list cs)
  | `StandardUniform -> Caml.Format.pp_print_string ppf "`StandardUniform"
  | `StandardGaussian -> Caml.Format.pp_print_string ppf "`StandardGaussian"

let format_low_level ~as_toplevel (ppf: Caml.Format.formatter) (type a) (c: a Code.low_level): unit =
  let open Code in
  let open Caml.Format in
  let pp_dims ppf dims =
    fprintf ppf "[|%a|]" (pp_print_list ~pp_sep:semi pp_print_int) @@ Array.to_list dims in
  let pp_indices ppf idcs =
    fprintf ppf "[|%a|]" (pp_print_list ~pp_sep:semi pp_print_int) @@
    Array.to_list @@ Array.map ~f:(function Shape.Symbol s -> s) idcs in
  let rec pp_ll: 'a. formatter -> 'a low_level -> unit = fun (ppf: formatter) (type a) (c: a low_level) ->
    (* FIXME: performance bug, bind the nodes [(get %d)] at the start of the program. *)
    match c with
    | Lines lines ->
      (pp_print_list ~pp_sep:semi pp_ll ppf @@ Array.to_list lines : unit)
    | For_loop {index=Symbol i; from_; to_; body} ->
      fprintf ppf "@[<2>for@ i%d = %d@ to %d@ do@ %a@]@ done" i from_ to_ pp_ll body
    | Value_at_node_id id ->
      fprintf ppf "(get %d).value" id
    | Gradient_at_node_id id ->
      fprintf ppf "(get %d).grad" id
    | LLCreate { tensor=Value_at_node_id id; precision=_; dims; init_op } ->
      fprintf ppf "@[<2>(get %d).value <-@ create_array@ %a %a@]" id pp_dims dims pp_print_init_op init_op
    | LLCreate { tensor=Gradient_at_node_id id; precision=_; dims; init_op } ->
      fprintf ppf "@[<2>(get %d).grad <-@ create_array@ %a %a@]" id pp_dims dims pp_print_init_op init_op
    | LLReset { tensor=Value_at_node_id id; precision=_; reset_op } ->
      fprintf ppf "@[<2>reset_array@ ((get %d).value) %a@]" id pp_print_init_op reset_op
    | LLReset { tensor=Gradient_at_node_id id; precision=_; reset_op } ->
      fprintf ppf "@[<2>reset_array@ ((get %d).grad) %a@]" id pp_print_init_op reset_op
    | Unoptimized_set (Value_at_node_id id, indices, v) ->
      fprintf ppf "@[<2>A.set (get %d).value@ %a@ %a@]" id pp_indices indices pp_ll v
    | Unoptimized_set (Gradient_at_node_id id, indices, v) ->
      fprintf ppf "@[<2>A.set (get %d).grad@ %a@ %a@]" id pp_indices indices pp_ll v
    | Unoptimized_get (Value_at_node_id id, indices) ->
      fprintf ppf "@[<2>A.get (get %d).value@ %a@]" id pp_indices indices
    | Unoptimized_get (Gradient_at_node_id id, indices) ->
      fprintf ppf "@[<2>A.get (get %d).grad@ %a@]" id pp_indices indices
      fprintf ppf "@[A.get (get %d).grad@ %a@]" id pp_indices indices
    | Unoptimized_binop (_op, _v1, _v2) -> fprintf ppf "()"
    | Unoptimized_unop (_op, _v) -> fprintf ppf "()"
    | Assign_routine (_routine, _proc) -> fprintf ppf "()" in
  fprintf ppf "@[<v>open Ocannl_runtime@ open Node@ ";
  fprintf ppf "@[<v>open Ocannl_runtime@ open Node@ open Float@ ";
  (match c with
   | Lines toplevel ->
     if as_toplevel then
       fprintf ppf "@[<2>let () =@ %a@]" (pp_print_list ~pp_sep:(fun p () -> fprintf p "@]@ @[<2>let () =@ ") pp_ll) @@
       Array.to_list toplevel
     else
      fprintf ppf "(@[<2>%a@]@,)" (pp_print_list ~pp_sep:semi pp_ll) @@
      Array.to_list toplevel

   | c -> pp_ll ppf c);
  fprintf ppf "@]"

let code_file_prefix = "nnrun"
let column_width = 100

(** Create a file to compile and later link. *)
let create_comp_unit compiled =
  let fname, oc =
    Caml.Filename.open_temp_file ~mode:[Open_wronly;Open_creat;Open_text]
      code_file_prefix ".ml" in
  (* FIXME(32): the following outputs truncated source code -- missing the last line:
  let ppf = Caml.Format.formatter_of_out_channel oc in
  Caml.Format.pp_set_geometry Caml.Format.str_formatter
    ~max_indent:(column_width/2) ~margin:column_width;
  let () = format_low_level ppf compiled in
  let () = Stdio.Out_channel.close oc in *)
  Caml.Format.pp_set_geometry Caml.Format.str_formatter
    ~max_indent:(column_width/2) ~margin:column_width;
  format_low_level ~as_toplevel:true Caml.Format.str_formatter compiled;
  let contents = Caml.Format.flush_str_formatter() in
  Stdio.Out_channel.output_string oc contents;
  Stdio.Out_channel.flush oc;
  Stdio.Out_channel.close oc;
  fname

let safe_remove fname =
  try Caml.Sys.remove fname with _ -> ()

let ocamlopt_path = "ocamlfind ocamlopt"

(** Compile the source file and make the .cmxs, returning its name and the name of the file with
    the compilation logs. *)
let compile_source ~with_debug src_fname =
  let basename = Caml.Filename.remove_extension src_fname in
  let log_fname = basename ^ ".log" in
  let plugin_fname = basename ^ ".cmxs" in
  let other_files = [basename ^ ".cmi"; basename ^ ".cmx"] in
  (* We need the byte objects directory in path because it contains the .cmi files. *)
  (* FIXME: un-hardcode the paths. *)
  let cmdline = ocamlopt_path ^ 
                " -I ~/ocannl/_build/default/lib -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/native -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/byte -package base -package stdio " ^
                " -shared"^(if with_debug then " -g" else "")^" -o " ^ plugin_fname ^
                " " ^ src_fname ^ " >> " ^ log_fname ^ " 2>&1" in
  (* TODO: consider using `Core` or `Core_unix`. *)
  let rc = Caml.Sys.command cmdline in
  while not @@ Caml.Sys.file_exists log_fname do () done;
  List.iter ~f:safe_remove other_files;
  plugin_fname, log_fname, rc

let first_file_span_re = Str.regexp @@
  code_file_prefix ^ {|[A-Za-z0-9]*\.ml\\?", line \([0-9]+\), characters \([0-9]+\)-\([0-9]+\)|}

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

let error_closing_delimiter = " #$}\027[0m "
let error_opening_delimiter =" \027[1;31m{$# "

let handle_error prefix ?formula ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace() in
  let exc_str = Caml.Printexc.to_string exc in
  let message =
    Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg prefix; msg exc_str; msg "\n"; msg backtrace;
  (match extra_error_msg with None -> () | Some extra ->
    msg "\nIn the context of:\n"; msg extra);
  let from_pos, to_pos = first_file_span ~contents ~message:(Buffer.contents message) in
  msg "\nIn code span ";
  msg error_opening_delimiter; msg "..."; msg error_closing_delimiter; msg ":\n";
  msg @@ String.sub contents ~pos:0 ~len:from_pos;
  msg error_opening_delimiter;
  msg @@ String.sub contents ~pos:from_pos ~len:(to_pos - from_pos);
  msg error_closing_delimiter;
  msg @@ String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos);
  let exc = Formula.Session_error (Buffer.contents message, formula) in
  Buffer.clear message;
  Stdio.prerr_endline @@ Option.value_exn (Formula.session_error_printer exc);
  raise exc

let load_native ?(with_debug=true) (prog: Code.program) =
  let compiled = emit prog in
  if not Dynlink.is_native then invalid_arg "ExecAsOCaml.load_forward: only works in native code";
  let source_fname = create_comp_unit compiled in
  Exn.protect ~finally:(fun () -> safe_remove source_fname)
    ~f:(fun () ->
        if with_debug then
          let contents =
            Caml.Format.pp_set_geometry Caml.Format.str_formatter
              ~max_indent:(column_width/2) ~margin:column_width;
            format_low_level ~as_toplevel:true Caml.Format.str_formatter compiled;
            Caml.Format.flush_str_formatter() in
          try
            let plugin_fname, log_fname, exitc = compile_source ~with_debug source_fname in
            let exec_logs = Stdio.In_channel.read_all log_fname in
            if exitc <> 0 then handle_error "Compilation error:\n" ~contents (Failure exec_logs)
            else
              Exn.protect ~finally:(fun () -> safe_remove plugin_fname; safe_remove log_fname)
                ~f:(fun () ->
                  try Dynlink.loadfile_private plugin_fname; Some contents
                  with 
                  | Dynlink.Error (Library's_module_initializers_failed exc) ->
                    handle_error "Runtime init error:\n" ~extra_error_msg:exec_logs ~contents exc
                  | exc ->
                    handle_error "Compilation or unknown runtime error:\n" ~extra_error_msg:exec_logs
                      ~contents exc)
          with 
          | Formula.Session_error _ as exc -> raise exc
          | exc -> handle_error "Compile-time error:\n" ~contents exc
        else (
          let plugin_fname, log_fname, exitc = compile_source ~with_debug source_fname in
          let exec_logs = Stdio.In_channel.read_all log_fname in
          if exitc <> 0 then handle_error "ExecAsOCaml.load_native: "
             ~contents:"<pass ~with_debug:true for debugging information>" (Failure exec_logs)
          else Dynlink.loadfile_private plugin_fname; None))