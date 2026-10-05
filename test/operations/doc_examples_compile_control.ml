(* The failure that motivated gh-ocannl-660: a removed accessor in selected Markdown must fail real
   typechecking. The neighboring record-field spelling proves the compiler/interface leg works; a
   failed compiler launch or missing include directory cannot pass as a negative result. *)
open Base
open Stdio
open Verdict.Claims
module E = Test_utils.Doc_examples_scan

let compile ~compiler ~interface code =
  let source = Stdlib.Filename.temp_file "doc_examples_control" ".ml" in
  let output = source ^ ".cmo" in
  let log = source ^ ".log" in
  (* CI may request colors; plain diagnostics must still name the rejected API contiguously. *)
  let environment =
    Unix.environment ()
    |> Array.filter ~f:(fun value -> not (String.is_prefix value ~prefix:"OCAML_COLOR="))
    |> fun inherited -> Array.append inherited [| "OCAML_COLOR=always" |]
  in
  Exn.protect
    ~f:(fun () ->
      let markdown = "```ocaml doc-check=control\n" ^ code ^ "\n```\n" in
      Out_channel.write_all source ~data:(E.render (E.parse ~path:"control.md" markdown));
      let fd = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
      let pid =
        Exn.protect
          ~f:(fun () ->
            Unix.create_process_env compiler
              [|
                compiler;
                "-color";
                "never";
                "-I";
                Stdlib.Filename.dirname interface;
                "-c";
                "-o";
                output;
                source;
              |]
              environment Unix.stdin fd fd)
          ~finally:(fun () -> Unix.close fd)
      in
      let _, status = Unix.waitpid [] pid in
      (status, In_channel.read_all log))
    ~finally:(fun () ->
      List.iter
        [ source; output; source ^ ".cmi"; log ]
        ~f:(fun path -> if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path))

(* The words of a compiler diagnostic line, its punctuation dropped. The error kind and the
   identifier are the contract; how the compiler delimits an identifier is presentation, and it
   changed between releases: OCaml 5.3 prints [Unbound value "Context.context"] where 5.4 and 5.5
   print it bare (gh-ocannl-1223). Matching the [Error:] line, not the whole log, keeps any other
   failure from passing: the log's source excerpt names the identifier for every error on its
   line. *)
let diagnostic_words line =
  String.split_on_chars line ~on:[ ' '; '\t'; ':'; '"'; '`'; '\'' ]
  |> List.filter ~f:(fun word -> not (String.is_empty word))

let () =
  let compiler = Stdlib.Sys.argv.(1) in
  let interface = Stdlib.Sys.argv.(2) in
  let positive, positive_log =
    compile ~compiler ~interface
      "let accessor (routine : Context.routine) = routine.Context.context"
  in
  if not (Poly.equal positive (Unix.WEXITED 0)) then eprintf "%s" positive_log;
  p "the extracted current record-field example compiles" (Poly.equal positive (Unix.WEXITED 0));
  let negative, negative_log =
    compile ~compiler ~interface
      "let accessor (routine : Context.routine) = Context.context routine"
  in
  let rejected_api =
    Poly.equal negative (Unix.WEXITED 2)
    && List.exists (String.split_lines negative_log) ~f:(fun line ->
        List.equal String.equal (diagnostic_words line)
          [ "Error"; "Unbound"; "value"; "Context.context" ])
  in
  if not rejected_api then eprintf "%s" negative_log;
  p "the stale accessor fails at name resolution even with ambient colors requested" rejected_api
