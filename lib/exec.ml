(* Included from BER MetaOCaml to avoid runtime build problems and to tweak. See:
   https://github.com/metaocaml/ber-metaocaml/blob/ber-n111/ber-metaocaml-111/runnative.ml *)
(* Given a closed code expression, compile it with the *native*
   compiler, link it in, and run returning
   its result or propagating raised exceptions.
*)

open Base

let load_path : string list ref = ref []

(* Add a directory to search for .cmo/.cmi files, needed
   for the sake of running the generated code .
   The specified directory is prepended to the load_path.
*)
let add_search_path : string -> unit = fun dir ->
  load_path := dir :: !load_path

let ocamlopt_path = 
  let open Caml.Filename in
  concat (concat (dirname (dirname (Config.standard_library))) 
   "bin") "ocamlfind ocamlopt"

(* Compile the source file and make the .cmxs, returning its name *)
let compile_source ~with_debug src_fname =
  let basename = Caml.Filename.remove_extension src_fname in
  let plugin_fname =  basename ^ ".cmxs" in
  let other_files  =  [basename ^ ".cmi"; basename ^ ".cmx";
                       basename ^ Config.ext_obj] in
  (* We need the byte objects directory in path because it contains the .cmi files. *)
  (* FIXME: un-hardcode the paths. *)
  let cmdline = ocamlopt_path ^ 
                " -I ~/ocannl/_build/default/lib -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/native -I ~/ocannl/_build/default/lib/.ocannl_runtime.objs/byte " ^
                " -shared"^(if with_debug then " -g" else "")^" -o " ^ plugin_fname ^
                (String.concat ~sep:"" @@ 
                 List.map ~f:(fun p -> " -I " ^ p) !load_path) ^
                " " ^ src_fname in      
  let rc = Caml.Sys.command cmdline in
  List.iter ~f:Caml.Sys.remove other_files;
  if rc = 0 then plugin_fname else 
    let () = Caml.Sys.remove plugin_fname in
    failwith "Ocannl.exec: .cmxs compilation failure"

(*
 Dynlink library can only load the unit and evaluate its top-level
 expressions. There is no provision for accessing the names defined 
 in the loaded unit. Therefore, the only way to get the result is
 to assign it to a reference cell defined in the main program.
 The Node module defines "result__" exactly for this purpose.

 Given the code cde, we generate a file

 Ocannl_runtime.Node.result__ := Some (Ocannl_runtime.Node.Obj.repr (cde))

which we then compile and link in.
*)

let code_file_prefix = "runn"

(* Create a file to compile and later link, using the given closed code *)
let create_comp_unit : 'a Codelib.closed_code -> string = fun cde ->
  let (fname,oc) =
    Caml.Filename.open_temp_file ~mode:[Open_wronly;Open_creat;Open_text]
      code_file_prefix ".ml" in
  let ppf = Caml.Format.formatter_of_out_channel oc in
  Caml.Format.pp_set_margin ppf 160;
  let ()  = Caml.Format.fprintf ppf
      "Ocannl_runtime.Node.result__ :=@ Some (Ocannl_runtime.Node.Obj.repr (@ %a))@."
      Codelib.format_code cde in
  (* FIXME: clean debug as needed. *)
  let ()  = Caml.Format.printf
      "Ocannl_runtime.Node.result__ :=@ Some (Ocannl_runtime.Node.Obj.repr (@ %a))@.%!"
      Codelib.format_code cde in
  let () = Stdio.Out_channel.close oc in
  fname                                 (* let the errors propagate *)


let run_native ?(with_debug=true) (type a) (cde: a Codelib.code): a =
  let closed = Codelib.close_code cde in
  if not Dynlink.is_native then
    failwith "run_native only works in the native code";
  let source_fname = create_comp_unit closed in
    let plugin_fname = compile_source ~with_debug source_fname in
  let () = Dynlink.loadfile_private plugin_fname in
  Caml.Sys.remove plugin_fname;
  Caml.Sys.remove source_fname;
  match !Ocannl_runtime.Node.result__ with
  | None -> assert false                (* can't happen *)
  | Some x -> 
      Ocannl_runtime.Node.result__ := None;                 (* prevent the memory leak *)
      Ocannl_runtime.Node.Obj.obj x
  (* If an exception is raised, leave the source and plug-in files,
     so to investigate the problem.
   *)
