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
  Exn.protect
    ~f:(fun () ->
      let markdown = "```ocaml doc-check=control\n" ^ code ^ "\n```\n" in
      Out_channel.write_all source ~data:(E.render (E.parse ~path:"control.md" markdown));
      let fd = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
      let pid =
        Exn.protect
          ~f:(fun () ->
            Unix.create_process compiler
              [| compiler; "-I"; Stdlib.Filename.dirname interface; "-c"; "-o"; output; source |]
              Unix.stdin fd fd)
          ~finally:(fun () -> Unix.close fd)
      in
      let _, status = Unix.waitpid [] pid in
      (status, In_channel.read_all log))
    ~finally:(fun () ->
      List.iter
        [ source; output; source ^ ".cmi"; log ]
        ~f:(fun path -> if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path))

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
  p "the extracted stale accessor fails specifically at OCaml name resolution"
    (Poly.equal negative (Unix.WEXITED 2)
    && String.is_substring negative_log ~substring:"Unbound value Context.context")
