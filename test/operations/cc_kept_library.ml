(* gh-ocannl-1247, gh-ocannl-1335: the cc backend keeps each compiled library in
   [build_files/<exe>/], beside the kernel's debug source, under a name unique to the process and
   the compile, whenever a library is asked for: under [output_dlls_in_build_directory=true] (the
   [cc_kept_library] rule) and under [output_debug_files_in_build_directory=true] alone (the
   [cc_kept_library-debug] rule), which used to leave it in the system temp directory for good. The
   compiler reads a private copy of the source, so a library named after that path landed in the
   system temp directory instead.

   A routine's name has no length limit and a file name has one, so the name carries at most
   [Utils.library_kernel_name_limit] bytes of it, cut at a character boundary and followed by a
   digest of the whole name: the test compiles a routine whose name alone nearly fills a file-name
   component, and reads a cut through two-byte characters off the naming function itself.

   Processes of one executable share that directory, and each one's startup empties it
   ([clean_up_build_files_on_startup] and [clean_up_log_files_on_startup], on by default, off in the
   test config, on in the children here). A running process holds a lock that tells a starting one
   it is there, and then the starting one removes nothing: it cannot tell the running process's
   ownerless artifacts from a finished run's, nor trust a gone-looking owner (the lock reaches
   across PID namespaces, liveness does not). So the test first plants the library of a process that
   is gone (an OS id no process has) and runs two children while it holds nothing: one compiles
   under debug files and exits, and the next one's startup removes both the debug source it left and
   the planted library. Then the test compiles (taking the lock), writes a log, plants another gone
   process's library, and relaunches itself as a child that compiles the same routine name. The
   child keeps a library of its own, under a name apart from this process's; its startup spares this
   process's library, log and (under debug files) debug source -- this process could be between
   linking that library and loading it, or still writing to that log -- and the planted library too.

   The libraries are read through [Test_utils.Generated.kept_libraries], which parses names with the
   backend's own [Utils.parse_kept_library_name], after [Generated.init] emptied this executable's
   directory, so a library found there was kept by this run. The children run no [Generated.init],
   which would sweep the parent's directory; their verdict output goes to stderr and their judgment
   reaches the parent as their exit status. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Generated = Test_utils.Generated

let routine = "gh1247_kept"
let stale_routine = "gh1335_stale"

(* Nearly a file-name component on its own: on top of it, the owner and compile fields of an uncut
   kept library's name would pass 255 bytes. *)
let long_routine = "gh1335_" ^ String.make 233 'k'
let mode = Stdlib.Sys.argv.(1)

let () =
  if not (List.mem [ "parent"; "child"; "stale"; "sweep" ] mode ~equal:String.equal) then
    invalid_arg ("cc_kept_library: unknown mode " ^ mode)

let compile_and_run routine =
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
let debug = Utils.settings.output_debug_files_in_build_directory

(* Under debug files the long-named routine's debug sources are named by its UNCUT name (only owned
   artifacts are cut): [long_routine] keeps [<name>-unoptimized.ll] within a 255-byte component, but
   on Windows the whole path is held to MAX_PATH (260 characters), which the build directory already
   spends a third of, so the first such write fails (gh-ocannl-1411). No length both exercises the
   cut and fits there; the [cc_kept_library] rule, without debug files, still compiles the long name
   on Windows. *)
let long_name_compiles = not (debug && Stdlib.Sys.win32)

(* A claim about the long-named routine: [claim label] where it compiles, a skip naming why where it
   cannot. *)
let long_claim label claim =
  Verdict.gate ~aggregation:`Environment ~when_:long_name_compiles
    ~on:"windows (debug file past MAX_PATH)" label (fun () -> claim label)

(* The child's flags: the parent's own configuration (which follows its mode argument), less the
   settings a child mode sets for itself. *)
let run_child child_mode settings =
  let exe = Stdlib.Sys.executable_name in
  let overridden arg =
    List.exists settings ~f:(fun setting ->
        String.is_prefix arg ~prefix:(String.lsplit2_exn setting ~on:'=' |> fst))
  in
  let args =
    Array.concat
      [
        [| exe; child_mode |];
        Array.filter
          (Array.sub Stdlib.Sys.argv ~pos:2 ~len:(Array.length Stdlib.Sys.argv - 2))
          ~f:(fun arg -> not (overridden arg));
        Array.of_list settings;
      ]
  in
  (* Flushed before the spawn, so the child cannot inherit and repeat buffered output. *)
  Stdio.Out_channel.flush Stdio.stdout;
  let child = Unix.create_process exe args Unix.stdin Unix.stderr Unix.stderr in
  match Unix.waitpid [] child with _, Unix.WEXITED 0 -> true | _ -> false

let clean_up_on_startup =
  [ "--ocannl_clean_up_build_files_on_startup=true"; "--ocannl_clean_up_log_files_on_startup=true" ]

let parent () =
  (* 1. This process has emptied the directory and holds no lock there yet (nothing here has named a
     path to write): what a finished process left is stale, and a starting process removes it. *)
  Generated.init ~backend_name:"cc";
  let departed_before = Generated.plant_kept_library ~pid:departed_pid routine in
  p "a finished process of this executable left a debug source"
    (run_child "stale"
       [
         "--ocannl_output_debug_files_in_build_directory=true";
         "--ocannl_clean_up_build_files_on_startup=false";
       ]);
  p "a process starting while no other runs ran" (run_child "sweep" clean_up_on_startup);
  p_none "its startup removed the finished process's debug source and a gone process's library"
    [ Generated.path ~ext:".c" stale_routine; departed_before ]
    ~f:Stdlib.Sys.file_exists;
  (* 2. The kept libraries. *)
  let backend, values = compile_and_run routine in
  Stdio.printf "backend: %s\n" backend;
  computes_3x_plus_1 "the routine computes 3x + 1 from its kept library" values;
  let long_kept =
    if long_name_compiles then
      let _backend, long_values = compile_and_run long_routine in
      Some (long_values, Generated.kept_libraries long_routine)
    else None
  in
  let with_long label f = long_claim label (fun label -> Option.iter long_kept ~f:(f label)) in
  with_long "a routine with a 240-byte name computes 3x + 1 from its kept library"
    (fun label (long_values, _) -> computes_3x_plus_1 label long_values);
  with_long "the long-named routine's library is kept, under its cut name"
    (fun label (_, long_kept) -> p label (List.length long_kept = 1));
  with_long "the long-named routine's library name fits a file-name component"
    (fun label (_, long_kept) ->
      p_all label long_kept ~f:(fun path -> String.length (Stdlib.Filename.basename path) <= 255));
  (* The digest in a cut name keeps apart routines whose names share their opening bytes. *)
  with_long "no library is found for another routine sharing the long name's opening bytes"
    (fun label (_, long_kept) ->
      p_empty label ~over:long_kept (Generated.kept_libraries ~owner:`Any (long_routine ^ "_other")));
  (* The cut never splits a character: a name of two-byte characters, placed so that the limit falls
     inside one. *)
  let wide_routine = "kernel" ^ String.concat (List.init 150 ~f:(fun _ -> "\u{e9}")) in
  let wide_name =
    Utils.kept_library_name ~pid:departed_pid ~nonce:"0badf00d" ~run_id:1 wide_routine
  in
  p "a kept library's name carries a whole-character prefix and a digest within the limit"
    (Option.exists (Utils.parse_kept_library_name wide_name) ~f:(fun { Utils.kernel_name; _ } ->
         String.length kernel_name <= Utils.library_kernel_name_limit
         && Stdlib.String.is_valid_utf_8 kernel_name
         &&
         match String.rsplit2 kernel_name ~on:'-' with
         | Some (prefix, digest) ->
             String.length digest = 8
             && String.length prefix >= Utils.library_kernel_name_limit - 10
             && String.is_prefix wide_routine ~prefix
         | None -> false));
  (* 3. This process's ownerless artifacts: a log, and under debug files the long-named routine's
     debug source, which no child writes. *)
  let log = Utils.diagn_log_file "gh1335_parent.log" in
  Stdio.Out_channel.write_all log ~data:"written by the parent\n";
  if debug then
    long_claim "this process wrote the long-named routine's debug source" (fun label ->
        p label (Stdlib.Sys.file_exists (Generated.path long_routine)));
  (* A library left by a process that is gone: while this process holds the lock, the child's
     startup cannot rule out that its owner is running where liveness cannot see. *)
  let departed_during = Generated.plant_kept_library ~run_id:2 ~pid:departed_pid routine in
  p "this process is seen as running" (Utils.process_alive (Utils.os_process_id ()));
  p "the stale library's process is seen as gone" (not (Utils.process_alive departed_pid));
  p "the child process compiled and ran the same routine name"
    (run_child "child" clean_up_on_startup);
  let own = Generated.kept_libraries routine in
  let departed = Generated.kept_libraries ~owner:(`Pid departed_pid) routine in
  let compiled =
    List.filter (Generated.kept_libraries ~owner:`Any routine) ~f:(fun f ->
        not (List.mem departed f ~equal:String.equal))
  in
  let others = List.filter compiled ~f:(fun f -> not (List.mem own f ~equal:String.equal)) in
  Stdio.eprintf "kept libraries (not part of the golden): %s\n%!"
    (String.concat ~sep:" " (compiled @ departed));
  p "this process's library stays in build_files through the child's startup cleanup"
    (List.length own = 1);
  p "the child kept its own library, under a name apart from this process's" (List.length others = 1);
  p "the child's startup kept the library naming a gone process too, while this process runs"
    (List.equal String.equal departed [ departed_during ]);
  p_all "each compiled library holds the routine's symbol" compiled ~min:2 ~f:(fun path ->
      String.is_substring (Stdio.In_channel.read_all path) ~substring:routine);
  p "this process's log stays through the child's startup cleanup" (Stdlib.Sys.file_exists log);
  if debug then
    long_claim "this process's debug source stays through the child's startup cleanup" (fun label ->
        p label (Stdlib.Sys.file_exists (Generated.path long_routine)))

let () =
  p "a library is kept: library output or debug files are requested"
    (debug || Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory");
  match mode with
  | "child" ->
      let _backend, values = compile_and_run routine in
      computes_3x_plus_1 "the child's routine computes 3x + 1 from its kept library" values
  | "stale" ->
      let _backend, values = compile_and_run stale_routine in
      computes_3x_plus_1 "the stale child's routine computes 3x + 1" values;
      p "the stale child's debug source is in place"
        (Stdlib.Sys.file_exists (Generated.path ~ext:".c" stale_routine))
  | "sweep" -> ()
  | _ -> parent ()
