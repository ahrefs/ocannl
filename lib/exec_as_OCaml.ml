open Base

let emit = Code.unoptimized_program

let pp_semi ppf () = Caml.Format.fprintf ppf ";@ "
let pp_symbol ppf (Shape.Symbol s) = Caml.Format.fprintf ppf "i%d" s

let pp_print_init_op ppf: Code.init_op -> unit = function
  | `Unspecified -> Caml.Format.pp_print_string ppf "`Unspecified"
  | `Constant_of_value c ->
    Caml.Format.fprintf ppf "(`Constant_of_value %f)" c
  | `Fixed_constant cs ->
    Caml.Format.(fprintf ppf "(`Fixed_constant @[<2>[|%a|]@])"
                   (pp_print_list ~pp_sep:pp_semi pp_print_float) @@ Array.to_list cs)
  | `Range_over_axis_from_end d ->
    Caml.Format.(fprintf ppf "(`Range_over_axis_from_end %d)" d)
  | `Range_over_offsets -> Caml.Format.(fprintf ppf "`Range_over_offsets")
  | `Standard_uniform -> Caml.Format.pp_print_string ppf "`Standard_uniform"
  | `Standard_gaussian -> Caml.Format.pp_print_string ppf "`Standard_gaussian"

let format_low_level ~as_toplevel (ppf: Caml.Format.formatter) (type a) (c: a Code.low_level): unit =
  let open Code in
  let open Caml.Format in
  let pp_dims ppf dims =
    fprintf ppf "[|%a|]" (pp_print_list ~pp_sep:pp_semi pp_print_int) @@ Array.to_list dims in
  let pp_indices ppf idcs =
    fprintf ppf "[|%a|]" (pp_print_list ~pp_sep:pp_semi pp_symbol) @@ Array.to_list idcs in
  let rec pp_ll: 'a. formatter -> 'a low_level -> unit = fun (ppf: formatter) (type a) (c: a low_level) ->
    (* FIXME: performance bug, bind the nodes [(get %d)] at the start of the program. *)
    match c with
    | Lines [||] -> fprintf ppf "()"
    | Lines lines -> (pp_print_list ~pp_sep:pp_semi pp_ll ppf @@ Array.to_list lines : unit)
    | For_loop {index=i; from_; to_; body} ->
      fprintf ppf "@[<2>for@ %a = %d@ to %d@ do@ %a@]@ done" pp_symbol i from_ to_ pp_ll body
    | Value_at_node_id id -> fprintf ppf "(get %d).value" id
    | Gradient_at_node_id id -> fprintf ppf "(get %d).grad" id
    | LLCreate { tensor=Value_at_node_id id; dims; init_op } ->
      fprintf ppf "@[<2>(get %d).value <-@ create_ndarray Single@ %a %a@]" id pp_dims dims pp_print_init_op init_op
    | LLCreate { tensor=Gradient_at_node_id id; dims; init_op } ->
      fprintf ppf "@[<2>(get %d).grad <-@ create_ndarray Single@ %a %a@]" id pp_dims dims pp_print_init_op init_op
    | LLReset { tensor=Value_at_node_id id; reset_op } ->
      fprintf ppf "@[<2>reset_ndarray@ %a@ ((get %d).value)@]" pp_print_init_op reset_op id
    | LLReset { tensor=Gradient_at_node_id id; reset_op } ->
      fprintf ppf "@[<2>reset_ndarray@ %a@ ((get %d).grad)@]" pp_print_init_op reset_op id
    | Unoptimized_set (Value_at_node_id id, indices, v) ->
      fprintf ppf "@[<2>set_from_float (get %d).value@ %a@ %a@]" id pp_indices indices pp_ll v
    | Unoptimized_set (Gradient_at_node_id id, indices, v) ->
      fprintf ppf "@[<2>set_from_float (get %d).grad@ %a@ %a@]" id pp_indices indices pp_ll v
    | Unoptimized_get (Value_at_node_id id, indices) ->
      fprintf ppf "@[<2>get_as_float (get %d).value@ %a@]" id pp_indices indices
    | Unoptimized_get (Gradient_at_node_id id, indices) ->
      fprintf ppf "@[<2>get_as_float (get %d).grad@ %a@]" id pp_indices indices
    | Unoptimized_binop (Skip_arg, _v1, v2) -> pp_ll ppf v2
    | Unoptimized_binop (Add, v1, v2) -> fprintf ppf "(@[<2>%a +@ %a@]@,)" pp_ll v1 pp_ll v2
    | Unoptimized_binop (Mul, v1, v2) -> fprintf ppf "(@[<2>%a *@ %a@]@,)" pp_ll v1 pp_ll v2
    | Unoptimized_binop (Relu_gate, v1, v2) ->
      fprintf ppf "(@[<2>if %a > 0.0@ then %a@ else 0.0@]@,)" pp_ll v1 pp_ll v2
    | Unoptimized_unop (Identity, v) -> pp_ll ppf v
    | Unoptimized_unop (Relu, v) ->
      fprintf ppf "(@[<2>let a = %a in@ if a > 0.0 then a else 0.0@]@,)" pp_ll v
    | Assign_routine ({node_id; field=`Forward}, proc) ->
      fprintf ppf "@[<2>(get %d).forward <-@ Some (@[<2>fun () ->@ %a@]@,)@]" node_id pp_ll proc
    | Assign_routine ({node_id; field=`Backprop}, proc) ->
      fprintf ppf "@[<2>(get %d).backprop <-@ Some (@[<2>fun () -> %a@]@,)@]" node_id pp_ll proc
    | Comment message -> fprintf ppf "(* %s *)()" message in
  fprintf ppf "@[<v>open Base@ open Ocannl_runtime@ open Node@ open Base.Float@ ";
  (match c with
   | Lines toplevel ->
     if as_toplevel then
       fprintf ppf "@[<2>let () =@ %a@]" (pp_print_list ~pp_sep:(fun p () -> fprintf p "@]@ @[<2>let () =@ ") pp_ll) @@
       Array.to_list toplevel
     else
      fprintf ppf "(@[<2>%a@]@,)" (pp_print_list ~pp_sep:pp_semi pp_ll) @@
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

let code_file_span_line = Str.regexp @@
  code_file_prefix ^ {|[A-Za-z0-9]*\.ml\\?", line \([0-9]+\), characters \([0-9]+\)-\([0-9]+\)|}
let code_file_span_lines = Str.regexp @@
  code_file_prefix ^ {|[A-Za-z0-9]*\.ml\\?", lines \([0-9]+\)-\([0-9]+\), characters \([0-9]+\)-\([0-9]+\)|}

(** Returns the character offset span inside [contents] corresponding to the first file span from [message].
    Returns [0, 0] if no span is found. *)
let first_file_span ~contents ~message =
  let last_char = String.length contents - 1 in
  try
    let multiline =
      try ignore(Str.search_forward code_file_span_line message 0); false
      with (Caml.Not_found | Not_found_s _) ->
        ignore(Str.search_forward code_file_span_lines message 0); true in
    let line_num = Int.of_string @@ Str.matched_group 1 message in
    let line_end = if multiline then Int.of_string @@ Str.matched_group 2 message else line_num in
    let char_start = Int.of_string @@ Str.matched_group (if multiline then 3 else 2) message in
    let char_end = Int.of_string @@ Str.matched_group (if multiline then 4 else 3) message in
    let rec line_offset ~line_num ~from =
      if line_num <= 1 then from else
        match String.index_from contents from '\n' with
        | None -> from
        | Some pos -> line_offset ~line_num:(line_num-1) ~from:(pos+1) in
    let line_from_offset = line_offset ~line_num ~from:0 in
    let line_end_offset = line_offset ~line_num:(line_end - line_num) ~from:line_from_offset in
    line_from_offset + char_start, line_end_offset + char_end
  with (Caml.Not_found | Not_found_s _) ->
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
  if not Dynlink.is_native then invalid_arg "Exec_as_OCaml.load_forward: only works in native code";
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
            if exitc <> 0 then
              let msg = handle_error "Compilation error:\n" ~contents (Failure exec_logs) in
              raise @@ Formula.Session_error (msg, None)
            else
              Exn.protect ~finally:(fun () -> safe_remove plugin_fname; safe_remove log_fname)
                ~f:(fun () ->
                  try Dynlink.loadfile_private plugin_fname; Some contents
                  with 
                  | Dynlink.Error (Library's_module_initializers_failed exc) ->
                    let msg =
                      handle_error "Runtime init error:\n" ~extra_error_msg:exec_logs ~contents exc in
                    raise @@ Formula.Session_error (msg, None)
                  | exc ->
                    let msg = handle_error "Compilation or unknown runtime error:\n"
                        ~extra_error_msg:exec_logs ~contents exc in
                    raise @@ Formula.Session_error (msg, None))
          with
          | Formula.Session_error _ as exc -> raise exc
          | exc ->
            let msg = handle_error "Compile-time error:\n" ~contents exc in
            raise @@ Formula.Session_error (msg, None)
        else (
          let plugin_fname, log_fname, exitc = compile_source ~with_debug source_fname in
          let exec_logs = Stdio.In_channel.read_all log_fname in
          if exitc <> 0 then
            let msg = handle_error "Exec_as_OCaml.load_native: "
             ~contents:"<pass ~with_debug:true for debugging information>" (Failure exec_logs) in
            raise @@ Formula.Session_error (msg, None)
          else Dynlink.loadfile_private plugin_fname; None))