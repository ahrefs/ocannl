let () =
  let usage_msg = "minised <regexp> <replacement> <input_filename> <output_filename>" in
  let regexp_str = ref "" in
  let replacement_str = ref "" in
  let input_filename = ref "" in
  let output_filename = ref "" in

  let speclist = [] in

  let anon_fun arg =
    if !regexp_str = "" then regexp_str := arg
    else if !replacement_str = "" then replacement_str := arg
    else if !input_filename = "" then input_filename := arg
    else if !output_filename = "" then output_filename := arg
    else raise (Arg.Bad ("Too many arguments: " ^ arg))
  in

  Arg.parse speclist anon_fun usage_msg;

  if !regexp_str = "" || !replacement_str = "" || !input_filename = "" || !output_filename = "" then (
    print_endline
      ("Provided arguments: <"
      ^ String.concat ">, <" [ !regexp_str; !replacement_str; !input_filename; !output_filename ]
      ^ ">");
    Arg.usage speclist usage_msg;
    exit 1);

  let ic = open_in !input_filename in
  let content =
    let rec read_lines acc =
      try
        let line = input_line ic in
        read_lines (line :: acc)
      with End_of_file ->
        close_in ic;
        List.rev acc
    in
    String.concat "\n" (read_lines [])
  in

  let re = Str.regexp !regexp_str in
  let new_content = Str.global_replace re !replacement_str content in

  let oc = open_out !output_filename in
  output_string oc new_content;
  close_out oc
