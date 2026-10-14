(* gh-ocannl-1247: under [output_dlls_in_build_directory=true] the cc backend keeps each compiled
   library in [build_files/<exe>/], beside the kernel's debug source, under a name unique to the
   process and the compile. The compiler reads a private temporary copy of the source (even under
   debug files), so a library named after that path landed in the system temp directory instead.

   Processes of one executable share that directory. The test relaunches itself as a child, which
   compiles the same routine name while the parent is still running and holds its own library; the
   child keeps a library of its own, under a name apart from the parent's. The child also runs the
   startup cleanup ([clean_up_build_files_on_startup], on by default, off in the test config), which
   empties the shared directory: it must spare the library of the running parent -- which could be
   between linking that library and loading it -- and still remove the library of a process that is
   gone (a fixture with an OS id no process has).

   The libraries are read through [Test_utils.Generated.kept_libraries], which parses names with the
   backend's own [Utils.parse_kept_library_name], after [Generated.init] emptied this executable's
   directory, so a library found there was kept by this run. The child runs no [Generated.init],
   which would sweep the parent's directory; its verdict output goes to stderr and its judgment
   reaches the parent as its exit status. *)

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

(* No process has this OS id: past every POSIX [pid_max], and odd, which a Windows process id never
   is. *)
let departed_pid = 2147483647

let () =
  p "library output is requested"
    (Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory");
  if child_mode then
    let _backend, values = compile_and_run () in
    computes_3x_plus_1 "the child's routine computes 3x + 1 from its kept library" values
  else (
    Generated.init ~backend_name:"cc";
    let backend, values = compile_and_run () in
    Stdio.printf "backend: %s\n" backend;
    computes_3x_plus_1 "the routine computes 3x + 1 from its kept library" values;
    (* A library left by a process that is gone, for the child's startup cleanup to find. *)
    Generated.plant_kept_library ~pid:departed_pid routine;
    p "this process is seen as running" (Utils.process_alive (Utils.os_process_id ()));
    p "the stale library's process is seen as gone" (not (Utils.process_alive departed_pid));
    let exe = Stdlib.Sys.executable_name in
    (* The parent's own configuration flags follow its mode argument; the child gets the same, and
       its startup cleanup on. *)
    let args =
      Array.concat
        [
          [| exe; "child" |];
          Array.sub Stdlib.Sys.argv ~pos:2 ~len:(Array.length Stdlib.Sys.argv - 2);
          [| "--ocannl_clean_up_build_files_on_startup=true" |];
        ]
    in
    (* Flushed before the spawn, so the child cannot inherit and repeat buffered output. *)
    Stdio.Out_channel.flush Stdio.stdout;
    let child = Unix.create_process exe args Unix.stdin Unix.stderr Unix.stderr in
    let _, status = Unix.waitpid [] child in
    p "the child process compiled and ran the same routine name"
      (match status with Unix.WEXITED 0 -> true | _ -> false);
    let own = Generated.kept_libraries routine in
    let every = Generated.kept_libraries ~owner:`Any routine in
    let others = List.filter every ~f:(fun f -> not (List.mem own f ~equal:String.equal)) in
    Stdio.eprintf "kept libraries (not part of the golden): %s\n%!" (String.concat ~sep:" " every);
    p "this process's library stays in build_files through the child's startup cleanup"
      (List.length own = 1);
    p "the child kept its own library, under a name apart from this process's"
      (List.length others = 1);
    p_empty "the child's startup cleanup removed the library of a process no longer running"
      ~over:every
      (Generated.kept_libraries ~owner:(`Pid departed_pid) routine);
    p_all "each kept library holds the routine's symbol" every ~min:2 ~f:(fun path ->
        String.is_substring (Stdio.In_channel.read_all path) ~substring:routine))
