(* gh-ocannl-1506: Shape.t is a private record, so the closed-shape guard's setters are the only way
   to write a shape. That is a property of the INTERFACE, which nothing executed can observe: this
   control hands real ocamlc programs against the built Shape interface. Reads and pattern matches
   must compile; an assignment to each mutable field, a record literal, a record update and the
   retired [t_of_sexp] must each be rejected with their own diagnostic. Every rejected leg compiled
   on the base, where the record was public. A failed compiler launch or a missing include directory
   cannot pass as a rejection: the positive leg runs through the same path, and each negative leg
   names the diagnostic it expects. *)
open Base
open Stdio
open Verdict.Claims

let compile ~compiler ~include_dirs code =
  let source = Stdlib.Filename.temp_file "shape_private_control" ".ml" in
  let output = Stdlib.Filename.chop_suffix source ".ml" ^ ".cmo" in
  let log = source ^ ".log" in
  Exn.protect
    ~f:(fun () ->
      Out_channel.write_all source ~data:code;
      let fd = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
      let pid =
        Exn.protect
          ~f:(fun () ->
            Unix.create_process compiler
              (Array.of_list
                 ([ compiler; "-color"; "never"; "-w"; "-a" ]
                 @ List.concat_map include_dirs ~f:(fun dir -> [ "-I"; dir ])
                 @ [ "-c"; "-o"; output; source ]))
              Unix.stdin fd fd)
          ~finally:(fun () -> Unix.close fd)
      in
      let _, status = Unix.waitpid [] pid in
      (status, In_channel.read_all log))
    ~finally:(fun () ->
      List.iter
        [ source; output; Stdlib.Filename.chop_suffix source ".ml" ^ ".cmi"; log ]
        ~f:(fun path -> if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path))

(* The words of the log's text, punctuation and quoting dropped: how a release delimits a name is
   presentation (gh-ocannl-1223). *)
let words text =
  String.split_on_chars text ~on:[ ' '; '\t'; '\n'; '\r'; ':'; '"'; '.'; ','; '('; ')' ]
  |> List.filter ~f:(Fn.non String.is_empty)

let contains_run text run =
  let rec go = function
    | [] -> false
    | _ :: rest as ws -> List.is_prefix ws ~prefix:run ~equal:String.equal || go rest
  in
  go (words text)

let prelude = "type shape = Ocannl_tensor.Shape.t\n"

let () =
  let compiler = Stdlib.Sys.argv.(1) in
  (* The directories of the interfaces the snippets reach: Shape's own, and Base's for its field
     types. *)
  let include_dirs =
    List.map (List.drop (Array.to_list Stdlib.Sys.argv) 2) ~f:Stdlib.Filename.dirname
  in
  let positive, positive_log =
    compile ~compiler ~include_dirs
      (prelude
     ^ "let read (sh : shape) = sh.closed && sh.id >= 0 && Option.is_none sh.discarded_by\n\
        let matched (sh : shape) = match sh with { closed; batch = _; _ } -> closed\n")
  in
  if not (Poly.equal positive (Unix.WEXITED 0)) then eprintf "%s" positive_log;
  p "reading and pattern-matching a shape's fields compiles" (Poly.equal positive (Unix.WEXITED 0));
  let rejected name code expected =
    let status, log = compile ~compiler ~include_dirs (prelude ^ code) in
    let ok = Poly.equal status (Unix.WEXITED 2) && contains_run log ("Error" :: expected) in
    if not ok then eprintf "%s:\n%s\n" name log;
    p name ok
  in
  List.iter
    [
      "batch";
      "input";
      "output";
      "batch_padding";
      "input_padding";
      "output_padding";
      "padding_elem";
      "closed";
      "discarded_by";
    ] ~f:(fun field ->
      rejected
        (Printf.sprintf "assigning the %s field is rejected as a private-type write" field)
        (Printf.sprintf "let write (sh : shape) = sh.%s <- sh.%s\n" field field)
        [ "Cannot"; "assign"; "field"; field; "of"; "the"; "private"; "type" ]);
  rejected "a record update of a shape is rejected as a private-type construction"
    "let copy (sh : shape) : shape = { sh with id = 0 }\n"
    [ "Cannot"; "create"; "values"; "of"; "the"; "private"; "type" ];
  rejected "a record literal of a shape is rejected as a private-type construction"
    "let fresh (sh : shape) : shape =\n\
    \  { batch = sh.batch; input = sh.input; output = sh.output; batch_padding = None;\n\
    \    input_padding = None; output_padding = None; padding_elem = None; closed = false;\n\
    \    discarded_by = None; id = 0; debug_name = \"forged\" }\n"
    [ "Cannot"; "create"; "values"; "of"; "the"; "private"; "type" ];
  rejected "deserializing a shape is no longer offered"
    "let parse = Ocannl_tensor.Shape.t_of_sexp\n"
    [ "Unbound"; "value"; "Ocannl_tensor"; "Shape"; "t_of_sexp" ]
