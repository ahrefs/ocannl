(* gh-ocannl-1268: a process's startup removes the temporary cc libraries of processes that are
   gone.

   A cc compile removes its temporary library once it is loaded (gh-ocannl-1197). Windows refuses to
   delete a loaded DLL, so there the library of a routine alive at exit stays in the temp directory.
   Its name records its owner ([Utils.temp_library_prefix]: pid and nonce, as a kept library's does
   since gh-ocannl-1247), and the startup of any later OCANNL process removes it once that owner is
   no longer running; the library of a running owner stays, since that process may be between
   linking it and loading it.

   The parent plants, in a private temp directory, the temporary library of a process that is gone
   (an OS id no process can have), one of its own (a running process), a library retained by debug
   files (whose name records no owner), and an unrelated library. It then starts a child with that
   directory as its temp directory; the child compiles and runs a routine. The child's startup must
   remove exactly the first. The child's own library is still loaded at its exit: on POSIX the
   compile already unlinked it, on Windows it stays, and a second child's startup must remove it --
   the Windows case end to end, gated to Windows where it is not vacuous.

   Each child's verdict output goes to stderr; its judgment reaches the parent as its exit
   status. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let routine = "gh1268_sweep"

let child_mode =
  match Stdlib.Sys.argv.(1) with
  | "child" -> true
  | "parent" -> false
  | mode -> invalid_arg ("cc_temp_library_sweep: unknown mode " ^ mode)

(* No process has this OS id: past every POSIX [pid_max], and odd, which a Windows process id never
   is. *)
let departed_pid = 2147483647

let child () =
  let ctx = Context.auto () in
  let x = TDSL.range 16 in
  let%op y = (5. *. x) + 2. in
  Train.set_materialized y.value;
  let comp = Train.forward y in
  let ctx, compiled = Context.compile ~name:routine ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx compiled in
  p "the child runs on cc" (String.equal (Context.backend_name ctx) "cc");
  p_alli "the child's routine computes 5x + 2" ~min:16
    (Array.to_list (Context.get_values ctx y.value))
    ~f:(fun i v -> Float.equal v (Float.of_int ((5 * i) + 2)));
  (* Alive at exit, so on Windows its library is still loaded when the process ends. *)
  ignore (Stdlib.Sys.opaque_identity compiled : Context.routine)

let parent () =
  let library_extension = Utils.dynamic_library_extension in
  let dir = Stdlib.Filename.temp_file "ocannl-gh1268-" "" in
  Stdlib.Sys.remove dir;
  Unix.mkdir dir 0o700;
  let present name = Stdlib.Sys.file_exists (Stdlib.Filename.concat dir name) in
  let listing () = Stdlib.Sys.readdir dir |> Array.to_list |> List.sort ~compare:String.compare in
  let plant_temp_library ~pid ~nonce =
    Stdlib.Filename.basename
      (Stdlib.Filename.temp_file ~temp_dir:dir
         (Utils.temp_library_prefix ~pid ~nonce ~run_id:1 routine)
         library_extension)
  in
  let departed = plant_temp_library ~pid:departed_pid ~nonce:"0badf00d" in
  let running = plant_temp_library ~pid:(Utils.os_process_id ()) ~nonce:(Utils.process_nonce ()) in
  (* The name shape of a library that [output_debug_files_in_build_directory] retains: the stem of
     the source's private copy, then the run id and [temp_file]'s random suffix. *)
  let retained = routine ^ "_1a2b3c_run_id_1_4d5e6f" ^ library_extension in
  let unrelated = "unrelated" ^ library_extension in
  List.iter [ retained; unrelated ] ~f:(fun f ->
      Stdio.Out_channel.write_all (Stdlib.Filename.concat dir f) ~data:f);
  let owner_of name = Utils.parse_temp_library_name name in
  p_all "the planted temporary libraries name their owners" [ departed; running ] ~f:(fun f ->
      Option.is_some (owner_of f));
  p_none "the retained and the unrelated library name no owner" [ retained; unrelated ] ~f:(fun f ->
      Option.is_some (owner_of f));
  p "this process is seen as running" (Utils.process_alive (Utils.os_process_id ()));
  p "the departed library's process is seen as gone" (not (Utils.process_alive departed_pid));
  let exe = Stdlib.Sys.executable_name in
  let args =
    Array.concat
      [
        [| exe; "child" |]; Array.sub Stdlib.Sys.argv ~pos:2 ~len:(Array.length Stdlib.Sys.argv - 2);
      ]
  in
  (* [Filename.get_temp_dir_name] reads TMPDIR on POSIX and TEMP on Windows; TMP is the other name
     Windows tools read. *)
  let temp_vars = [ "TMPDIR"; "TEMP"; "TMP" ] in
  let env =
    Array.append
      (Array.filter (Unix.environment ()) ~f:(fun kv ->
           not
             (List.exists temp_vars ~f:(fun var ->
                  String.is_prefix (String.uppercase kv) ~prefix:(var ^ "=")))))
      (Array.of_list (List.map temp_vars ~f:(fun var -> var ^ "=" ^ dir)))
  in
  let run_child () =
    (* Flushed before the spawn, so the child cannot inherit and repeat buffered output. *)
    Stdio.Out_channel.flush Stdio.stdout;
    let pid = Unix.create_process_env exe args env Unix.stdin Unix.stderr Unix.stderr in
    match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false
  in
  let planted = [ departed; running; retained; unrelated ] in
  Exn.protect
    ~finally:(fun () ->
      List.iter (listing ()) ~f:(fun f ->
          try Stdlib.Sys.remove (Stdlib.Filename.concat dir f) with Sys_error _ -> ());
      try Unix.rmdir dir with Unix.Unix_error _ -> ())
    ~f:(fun () ->
      p "the first child compiled and ran its routine" (run_child ());
      let after_first = listing () in
      Stdio.eprintf "temp directory after the first child (not part of the golden): %s\n%!"
        (String.concat ~sep:" " after_first);
      p "the child's startup removed the temporary library of a process no longer running"
        (not (present departed));
      p "the child's startup kept the temporary library of a running process" (present running);
      p_all "the child's startup left the retained and the unrelated library alone"
        [ retained; unrelated ] ~f:present;
      let first_child_left =
        List.filter after_first ~f:(fun f ->
            (not (List.mem planted f ~equal:String.equal)) && Option.is_some (owner_of f))
      in
      gated_exists ~aggregation:`Environment ~when_:Sys.win32
        ~on:"POSIX, which removes a loaded library at once"
        "the first child's loaded library outlived it, named after its owner" first_child_left
        ~f:(fun f ->
          Option.exists (owner_of f) ~f:(fun { Utils.kernel_name; owner_pid; _ } ->
              String.equal kernel_name routine && owner_pid <> Utils.os_process_id ()));
      p "the second child compiled and ran its routine" (run_child ());
      gated_all ~aggregation:`Environment ~when_:Sys.win32
        ~on:"POSIX, which removes a loaded library at once"
        "the second child's startup removed every library the first child left" first_child_left
        ~f:(fun f -> not (present f));
      p "the running process's library survived the second child's startup too" (present running))

let () = if child_mode then child () else parent ()
