open Base

let keep_files_in_run_directory = ref false
let emit = Code.to_low_level_program
let pp_semi ppf () = Caml.Format.fprintf ppf ";@ "
let pp_symbol ppf (Shape.Symbol s) = Caml.Format.fprintf ppf "i%d" s

let pp_symbolic_index ?provider_dim ppf =
  let open Shape in
  function
  | Iterator (Symbol s) | Dynamic_recipient (Symbol s) -> Caml.Format.fprintf ppf "i%d" s
  | Fixed_idx i -> Caml.Format.fprintf ppf "%d" i
  | Dynamic_provider _ -> Caml.Format.fprintf ppf "%d" @@ Option.value_exn provider_dim

let pp_print_init_op ppf : Code.init_op -> unit = function
  | Constant_fill cs ->
      Caml.Format.(
        fprintf ppf "(Constant_fill @[<2>[|%a|]@])" (pp_print_list ~pp_sep:pp_semi pp_print_float)
        @@ Array.to_list cs)
  | Range_over_offsets -> Caml.Format.(fprintf ppf "Range_over_offsets")
  | Standard_uniform -> Caml.Format.pp_print_string ppf "Standard_uniform"

let format_low_level ~as_toplevel (ppf : Caml.Format.formatter) (type a) (c : a Code.low_level) : unit =
  let open Code in
  let open Caml.Format in
  let pp_indices ?provider_dim ppf idcs =
    fprintf ppf "[|%a|]" (pp_print_list ~pp_sep:pp_semi (pp_symbolic_index ?provider_dim))
    @@ Array.to_list idcs
  in
  let pp_idcs = pp_indices ?provider_dim:None in
  let rec pp_ll : 'a. formatter -> 'a low_level -> unit =
   fun (ppf : formatter) (type a) (c : a low_level) ->
    (* FIXME: performance bug, bind the nodes [(get %d)] at the start of the program. *)
    match c with
    | Lines [||] -> fprintf ppf "()"
    | Lines lines -> (pp_print_list ~pp_sep:pp_semi pp_ll ppf @@ Array.to_list lines : unit)
    | For_loop { index = i; from_; to_; body } ->
        fprintf ppf "@[<2>for@ %a = %d@ to %d@ do@ %a@]@ done" pp_symbol i from_ to_ pp_ll body
    | Value_at_node_id id -> fprintf ppf "(get %d).value" id
    | Gradient_at_node_id id -> fprintf ppf "(get_form %d).grad" id
    | Fill { tensor = Value_at_node_id id; value } ->
        fprintf ppf "@[<2>fill_from_float (get %d).value@ (%a)@]" id pp_ll value
    | Fill { tensor = Gradient_at_node_id id; value } ->
        fprintf ppf "@[<2>fill_from_float (get_form %d).grad@ (%a)@]" id pp_ll value
    | Set (Value_at_node_id id, indices, v) ->
        fprintf ppf "@[<2>set_from_float (get %d).value@ (%a)@ (%a)@]" id pp_idcs indices pp_ll v
    | Set (Gradient_at_node_id id, indices, v) ->
        fprintf ppf "@[<2>set_from_float (get_form %d).grad@ (%a)@ (%a)@]" id pp_idcs indices pp_ll v
    | Dynamic_indices { tensor = Value_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices ("(get " ^ Int.to_string id ^ ").value") ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Dynamic_indices { tensor = Gradient_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices
          ("(get_form " ^ Int.to_string id ^ ").grad")
          ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Get (Value_at_node_id id, indices) ->
        fprintf ppf "@[<2>get_as_float (get %d).value@ (%a)@]" id pp_idcs indices
    | Constant c -> fprintf ppf "(%f)" c
    | Get (Gradient_at_node_id id, indices) ->
        fprintf ppf "@[<2>get_as_float (get_form %d).grad@ (%a)@]" id pp_idcs indices
    | Binop (Arg1, v1, _v2) -> pp_ll ppf v1
    | Binop (Arg2, _v1, v2) -> pp_ll ppf v2
    | Binop (Add, v1, v2) -> fprintf ppf "(@[<2>(%a) +@ (%a)@]@,)" pp_ll v1 pp_ll v2
    | Binop (Mul, v1, v2) -> fprintf ppf "(@[<2>(%a) *@ (%a)@]@,)" pp_ll v1 pp_ll v2
    | Binop (ToPowOf, v1, v2) ->
        (* if is_integer p then fprintf ppf "(@[<2>int_pow (%a) (%d)@]@,)" pp_ll v1 (to_int p) *)
        fprintf ppf "(@[<2>(%a) **@ (%a)@]@,)" pp_ll v1 pp_ll v2
    | Binop (Relu_gate, v1, v2) -> fprintf ppf "(@[<2>if %a > 0.0@ then %a@ else 0.0@]@,)" pp_ll v1 pp_ll v2
    | Unop (Identity, v) -> pp_ll ppf v
    | Unop (Relu, v) -> fprintf ppf "(@[<2>let a = %a in@ if a > 0.0 then a else 0.0@]@,)" pp_ll v
    | Comment message -> fprintf ppf "(* %s *)()" message
  and dynamic_indices tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    Array.iteri dynamic_idcs ~f:(fun provider_dim sym ->
        fprintf ppf "let@ %a = Int.(@[<2>(get_as_int %s@ (%a)) %% %d@]) in@ " pp_symbol sym tensor
          (pp_indices ~provider_dim) tensor_idcs target_dims.(provider_dim));
    pp_ll ppf body
  in
  (match c with
  | Lines toplevel ->
      if as_toplevel then
        fprintf ppf "@[<2>let () =@ %a@]"
          (pp_print_list ~pp_sep:(fun p () -> fprintf p "@]@ @[<2>let () =@ ") pp_ll)
        @@ Array.to_list toplevel
      else fprintf ppf "(@[<2>%a@]@,)" (pp_print_list ~pp_sep:pp_semi pp_ll) @@ Array.to_list toplevel
  | c -> pp_ll ppf c);
  fprintf ppf "@]"

let format_ll_prog (ppf : Caml.Format.formatter) (p : Code.low_level_program) : unit =
  let open Code in
  let open Caml.Format in
  fprintf ppf "@[<v>open Base@ open Ocannl_runtime@ open Node@ open Base.Float@ ";
  match p with
  | Perform proc -> format_low_level ~as_toplevel:true ppf proc
  | Assign_routine ({ id; field = `Forward }, proc) ->
      fprintf ppf "@[<2>let () = (get_form %d).forward :=@ Some (@[<2>fun () ->@ %a@]@,)@]" id
        (format_low_level ~as_toplevel:false)
        proc
  | Assign_routine ({ id; field = `Backprop }, proc) ->
      fprintf ppf "@[<2>let () = (get_form %d).backprop :=@ Some (@[<2>fun () -> %a@]@,)@]" id
        (format_low_level ~as_toplevel:false)
        proc
  | Assign_suspension proc ->
      fprintf ppf "@[<2>let () = most_recent_suspension@ := Some (@[<2>fun () -> %a@]@,)@]"
        (format_low_level ~as_toplevel:false)
        proc
  | Assign_session_prepare_step proc ->
      fprintf ppf "@[<2>let () = global.session_prepare_step@ := Some (@[<2>fun () -> %a@]@,)@]"
        (format_low_level ~as_toplevel:false)
        proc

let column_width = 100
let unique_id = ref @@ ((Int63.to_int_trunc @@ Time_now.nanoseconds_since_unix_epoch ()) % 1000) * 100
let safe_remove fname = try Caml.Sys.remove fname with _ -> ()

(** Create a file to compile and later link. *)
let create_comp_unit ~name compiled =
  let f_name =
    if !Code.keep_files_in_run_directory then name ^ ".ml" else Caml.Filename.temp_file (name ^ "_") ".ml"
  in
  safe_remove f_name;
  let oc = Out_channel.open_text f_name in
  (* FIXME(#32): the following outputs truncated source code -- missing the last line: *
     let ppf = Caml.Format.formatter_of_out_channel oc in
     Caml.Format.pp_set_geometry Caml.Format.str_formatter
     ~max_indent:(column_width/2) ~margin:column_width;
     let () = format_low_level ~as_toplevel:true ppf compiled in
     let () = Stdio.Out_channel.close oc in
     let () = Stdio.printf "FIXME(32): file content:\n%s\nEND file content\n%!"
     (Stdio.In_channel.read_all fname) in
   * Defensive variant: *)
  Caml.Format.pp_set_geometry Caml.Format.str_formatter ~max_indent:(column_width / 2) ~margin:column_width;
  format_ll_prog Caml.Format.str_formatter compiled;
  let contents = Caml.Format.flush_str_formatter () in
  Stdio.Out_channel.output_string oc contents;
  Stdio.Out_channel.flush oc;
  Stdio.Out_channel.close oc;
  f_name

let ocamlopt_path = "ocamlfind ocamlopt"

(** Compile the source file and make the .cmxs, returning its name and the name of the file with
    the compilation logs. *)
let compile_source src_fname =
  let basename = Caml.Filename.remove_extension src_fname in
  let log_fname = basename ^ ".log" in
  let plugin_fname = basename ^ ".cmxs" in
  let other_files = [ basename ^ ".cmi"; basename ^ ".cmx"; basename ^ ".o" ] in
  (* We need the byte objects directory in path because it contains the .cmi files. *)
  (* FIXME: un-hardcode the paths. *)
  safe_remove plugin_fname;
  let cmdline =
    ocamlopt_path
    ^ " -I ~/ocannl/_build/default/lib -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/native -I \
       ~/ocannl/_build/default/lib/.ocannl_runtime.objs/byte -package base -package stdio " ^ " -shared"
    ^ (if !Code.with_debug then " -g" else "")
    ^ " -o " ^ plugin_fname ^ " " ^ src_fname ^ " >> " ^ log_fname ^ " 2>&1"
  in
  (* TODO: consider using `Core` or `Core_unix`. *)
  let rc =  Caml.Sys.command cmdline in
  while not @@ Caml.Sys.file_exists log_fname do
    ()
  done;
  List.iter ~f:safe_remove other_files;
  (plugin_fname, log_fname, rc)

let code_file_span_line ~name =
  Str.regexp @@ name ^ {|[_A-Za-z0-9]*\.ml\\?", line \([0-9]+\), characters \([0-9]+\)-\([0-9]+\)|}

let code_file_span_lines ~name =
  Str.regexp @@ name ^ {|[_A-Za-z0-9]*\.ml\\?", lines \([0-9]+\)-\([0-9]+\), characters \([0-9]+\)-\([0-9]+\)|}

(** Returns the character offset span inside [contents] corresponding to the first file span from [message].
    Returns [0, 0] if no span is found. *)
let first_file_span ~name ~contents ~message =
  let last_char = String.length contents - 1 in
  try
    let multiline =
      try
        ignore (Str.search_forward (code_file_span_line ~name) message 0);
        false
      with Caml.Not_found | Not_found_s _ ->
        ignore (Str.search_forward (code_file_span_lines ~name) message 0);
        true
    in
    let line_num = Int.of_string @@ Str.matched_group 1 message in
    let line_end = if multiline then Int.of_string @@ Str.matched_group 2 message else line_num in
    let char_start = Int.of_string @@ Str.matched_group (if multiline then 3 else 2) message in
    let char_end = Int.of_string @@ Str.matched_group (if multiline then 4 else 3) message in
    let rec line_offset ~line_num ~from =
      if line_num <= 1 then from
      else
        match String.index_from contents from '\n' with
        | None -> from
        | Some pos -> line_offset ~line_num:(line_num - 1) ~from:(pos + 1)
    in
    let line_from_offset = line_offset ~line_num ~from:0 in
    let line_end_offset = line_offset ~line_num:(line_end - line_num) ~from:line_from_offset in
    (line_from_offset + char_start, line_end_offset + char_end)
  with Caml.Not_found | Not_found_s _ -> (0, last_char)

let error_closing_delimiter = " #$}\027[0m "
let error_opening_delimiter = " \027[1;31m{$# "

let error_message ~name ~prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace () in
  let exc_str = Caml.Printexc.to_string exc in
  let message = Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg prefix;
  msg exc_str;
  msg "\n";
  msg backtrace;
  (match extra_error_msg with
  | None -> ()
  | Some extra ->
      msg "\nIn the context of:\n";
      msg extra);
  let from_pos, to_pos = first_file_span ~name ~contents ~message:(Buffer.contents message) in
  (* Be defensive to avoid introducing a misleading error. *)
  let from_pos = Int.min from_pos @@ (String.length contents - 1) in
  let to_pos = Int.min to_pos @@ (String.length contents - 1) in
  msg "\nIn code span ";
  msg error_opening_delimiter;
  msg "...";
  msg error_closing_delimiter;
  msg ":\n";
  msg @@ String.sub contents ~pos:0 ~len:from_pos;
  msg error_opening_delimiter;
  msg @@ String.sub contents ~pos:from_pos ~len:(to_pos - from_pos);
  msg error_closing_delimiter;
  msg @@ String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos);
  Buffer.contents message

let load_native (prog : Code.program) =
  let compiled = emit prog in
  let name = Code.get_name prog ^ "_u" ^ Int.to_string (Int.incr unique_id; !unique_id) in
  if not Dynlink.is_native then invalid_arg "Exec_as_OCaml.load_forward: only works in native code";
  let source_fname = create_comp_unit ~name compiled in
  Exn.protect
    ~finally:(fun () -> if not !Code.keep_files_in_run_directory then safe_remove source_fname)
    ~f:(fun () ->
      if !Code.with_debug then
        (* TODO: don't generate the source twice. *)
        let contents =
          Caml.Format.pp_set_geometry Caml.Format.str_formatter ~max_indent:(column_width / 2)
            ~margin:column_width;
          format_ll_prog Caml.Format.str_formatter compiled;
          Caml.Format.flush_str_formatter ()
        in
        try
          let plugin_fname, log_fname, exitc = compile_source source_fname in
          let exec_logs = Stdio.In_channel.read_all log_fname in
          if exitc <> 0 then
            Formula.handle_error
            @@ error_message ~name ~prefix:"Compilation error:\n" ~contents (Failure exec_logs)
          else
            Exn.protect
              ~finally:(fun () ->
                safe_remove plugin_fname;
                safe_remove log_fname)
              ~f:(fun () ->
                try
                  Dynlink.loadfile_private plugin_fname;
                  Some contents
                with
                | Dynlink.Error (Library's_module_initializers_failed exc) ->
                    Formula.handle_error
                    @@ error_message ~name ~prefix:"Runtime init error:\n" ~extra_error_msg:exec_logs
                         ~contents exc
                | exc ->
                    Formula.handle_error
                    @@ error_message ~name ~prefix:"Compilation or unknown runtime error:\n"
                         ~extra_error_msg:exec_logs ~contents exc)
        with
        | Formula.Session_error _ as exc -> raise exc
        | exc -> Formula.handle_error @@ error_message ~name ~prefix:"Compile-time error:\n" ~contents exc
      else
        let plugin_fname, log_fname, exitc = compile_source source_fname in
        let exec_logs = Stdio.In_channel.read_all log_fname in
        if exitc <> 0 then
          Formula.handle_error
          @@ error_message ~name ~prefix:"Exec_as_OCaml.load_native: "
               ~contents:"<Set `Code.CDSL.with_debug : =true` for debugging information>" (Failure exec_logs)
        else Dynlink.loadfile_private plugin_fname;
        None)
