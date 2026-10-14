(* gh-ocannl-1247: under [output_dlls_in_build_directory=true] the cc backend keeps each compiled
   library in [build_files/<exe>/], beside the kernel's debug source, under a name unique to the
   process and the compile -- and two concurrently running processes of one executable, which share
   that directory, each keep their own.

   The compiler reads a private temporary copy of the source (even under debug files), so a library
   named after that path lands in the system temp directory; that is where it used to go. The test
   reads the kept library through [Test_utils.Generated.kept_libraries], which resolves names with
   the backend's own naming function, after [Generated.init] emptied this executable's directory --
   so a library found there was kept by this run.

   The concurrency leg relaunches this executable as a child, which compiles the same routine name
   while the parent compiles it too. Two libraries afterwards, one per pid, is what a cross-process
   collision would break: a shared name would leave one file, the later compile's, which the earlier
   process's [dlopen] could have read half-written. The child runs no [Generated.init], which would
   sweep the parent's directory; its verdict output goes to stderr and its judgment reaches the
   parent as its exit status. Where they actually overlap in time is the scheduler's business; the
   names do not depend on it. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Generated = Test_utils.Generated

let routine = "gh1247_kept"

let child_mode =
  match Stdlib.Sys.argv.(1) with
  | "child" -> true
  | "parent" -> false
  | mode -> invalid_arg ("cc_kept_library: unknown mode " ^ mode)

let compile_and_run () =
  let ctx = Context.auto () in
  let x = TDSL.range 16 in
  let%op y = (3. *. x) + 1. in
  Train.set_materialized y.value;
  let comp = Train.forward y in
  let ctx, compiled = Context.compile ~name:routine ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx compiled in
  (Context.backend_name ctx, Context.get_values ctx y.value)

let computes_3x_plus_1 claim values =
  p_alli claim (Array.to_list values) ~min:16 ~f:(fun i v ->
      Float.equal v (Float.of_int ((3 * i) + 1)))

let () =
  p "library output is requested"
    (Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory");
  if child_mode then
    let _backend, values = compile_and_run () in
    computes_3x_plus_1 "the child's routine computes 3x + 1 from its kept library" values
  else (
    Generated.init ~backend_name:"cc";
    let exe = Stdlib.Sys.executable_name in
    (* The parent's own configuration flags follow its mode argument; the child gets the same. *)
    let args =
      Array.append [| exe; "child" |]
        (Array.sub Stdlib.Sys.argv ~pos:2 ~len:(Array.length Stdlib.Sys.argv - 2))
    in
    (* Flushed before the spawn, so the child cannot inherit and repeat buffered output. *)
    Stdio.Out_channel.flush Stdio.stdout;
    let child = Unix.create_process exe args Unix.stdin Unix.stderr Unix.stderr in
    let backend, values = compile_and_run () in
    let _, status = Unix.waitpid [] child in
    Stdio.printf "backend: %s\n" backend;
    computes_3x_plus_1 "the routine computes 3x + 1 from its kept library" values;
    p "the concurrent child process compiled and ran the same routine name"
      (match status with Unix.WEXITED 0 -> true | _ -> false);
    let own = Generated.kept_libraries routine in
    let child_libraries = Generated.kept_libraries ~pid:child routine in
    Stdio.eprintf "kept libraries (not part of the golden): %s\n%!"
      (String.concat ~sep:" " (own @ child_libraries));
    p "this process kept exactly one library for the routine, in build_files" (List.length own = 1);
    p "the concurrent child kept its own library, apart from this process's"
      (List.length child_libraries = 1);
    p_all "each kept library holds the routine's symbol" (own @ child_libraries) ~min:2
      ~f:(fun path -> String.is_substring (Stdio.In_channel.read_all path) ~substring:routine))
