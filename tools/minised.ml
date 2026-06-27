let () =
  let usage_msg =
    "minised [--normalize-eol] <regexp> <replacement> <input_filename> <output_filename>\n\
     minised --normalize-eol-only <input_filename> <output_filename>"
  in
  let normalize_eol = ref false in
  let normalize_eol_only = ref false in
  let args = ref [] in

  let speclist =
    [
      ( "--normalize-eol",
        Arg.Set normalize_eol,
        "Normalize CRLF line endings to LF before writing output." );
      ( "--normalize-eol-only",
        Arg.Set normalize_eol_only,
        "Only normalize CRLF line endings to LF; do not perform regexp replacement." );
    ]
  in

  let anon_fun arg = args := arg :: !args in

  Arg.parse speclist anon_fun usage_msg;

  let args = List.rev !args in
  let usage_error () =
    print_endline ("Provided arguments: <" ^ String.concat ">, <" args ^ ">");
    Arg.usage speclist usage_msg;
    exit 1
  in
  let regexp_str, replacement_str, input_filename, output_filename =
    match (!normalize_eol_only, args) with
    | true, [ input_filename; output_filename ] -> (None, None, input_filename, output_filename)
    | true, _ -> usage_error ()
    | false, [ regexp_str; replacement_str; input_filename; output_filename ] ->
        (Some regexp_str, Some replacement_str, input_filename, output_filename)
    | false, _ -> usage_error ()
  in

  let ic = open_in_bin input_filename in
  let content =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> really_input_string ic (in_channel_length ic))
  in
  (* Generated backend sources and log files are compared as text goldens, so callers may opt into
     LF output even when the host writes CRLF. *)
  let content =
    if !normalize_eol || !normalize_eol_only then
      Str.global_replace (Str.regexp "\r\n") "\n" content
    else content
  in

  let new_content =
    match (regexp_str, replacement_str) with
    | Some regexp_str, Some replacement_str ->
        let re = Str.regexp regexp_str in
        Str.global_replace re replacement_str content
    | None, None -> content
    | _ -> assert false
  in

  let oc = open_out_bin output_filename in
  output_string oc new_content;
  close_out oc
