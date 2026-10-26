(* gh-ocannl-1636: under [debug_log_to_stream_files=true] a logged routine's lines reach its stream
   file and nothing reaches stdout, with no [Utils.capture_stdout_logs] window around the run.

   The rule runs this with routine logging and the stream-file key on. CUDA and HIP log with
   device-side [printf], which their runtimes write to process stdout at a synchronization point;
   before gh-ocannl-1636 only a caller's capture window handed those lines to the log processor that
   honours the key, so a run outside one printed them on stdout. Now the backend opens the window
   around each logged launch itself. The C backends log to files and never touched stdout, so there
   the claims hold trivially; the backend that ran is printed on stderr, since the golden is the
   same on every backend. On Metal a logged routine can crash the host process once it runs
   (gh-ocannl-1594), and its logs go to os_log rather than a stream file, so there the routine only
   compiles and the claims about its run are skipped. *)

open Base
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* [f]'s result and every byte written to file descriptor 1 while it ran, by OCaml and C code alike:
   the descriptor itself is routed to a temporary file, as device [printf] writes to it. Both stdio
   layers are flushed on each side of the redirection, and fd 1 is restored on every path. *)
let stdout_of f =
  let file = Stdlib.Filename.temp_file "routine_log_stream_files" ".txt" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove file with Sys_error _ -> ())
    ~f:(fun () ->
      let flush () =
        Stdio.Out_channel.flush Stdio.stdout;
        Utils.flush_c_streams ()
      in
      flush ();
      let saved = Unix.dup Unix.stdout in
      let result =
        Exn.protect
          ~finally:(fun () ->
            flush ();
            Unix.dup2 saved Unix.stdout;
            Unix.close saved)
          ~f:(fun () ->
            let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
            Exn.protect ~finally:(fun () -> Unix.close fd) ~f:(fun () -> Unix.dup2 fd Unix.stdout);
            f ())
      in
      (result, Stdio.In_channel.read_all file))

let n = 4

(* The descriptors this process holds open, where the host lists them. *)
let fd_dir = "/dev/fd"
let open_descriptors () = Array.length (Stdlib.Sys.readdir fd_dir)

let () =
  Stdio.eprintf "routine_log_stream_files: backend %s (not part of the golden)\n%!" backend_name;
  p "the run logs its routines to stream files"
    (Utils.debug_log_from_routines ()
    && Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files");
  (* Each logged CUDA or HIP launch now opens a capture window of its own, so a window must pass
     through every non-log line written inside it and give back every descriptor it opens, however
     soon its body returns. Before gh-ocannl-1636 the reader's pipe end and its duplicate of stdout
     leaked on the normal path, and a body that returned before the reader had started closed the
     descriptor the reader was about to duplicate, so the window lost its lines. Checked through
     [capture_stdout_logs] itself, on every backend. *)
  let windows = 5 in
  let line k = "window " ^ Int.to_string k ^ " passes this line through" in
  let lists_descriptors = Stdlib.Sys.file_exists fd_dir in
  let before = if lists_descriptors then open_descriptors () else 0 in
  let (), passed =
    stdout_of (fun () ->
        for k = 1 to windows do
          Utils.capture_stdout_logs (fun () -> Stdio.print_endline (line k))
        done)
  in
  let leaked = if lists_descriptors then open_descriptors () - before else 0 in
  if leaked <> 0 then Stdio.eprintf "routine_log_stream_files: %d descriptors leaked\n%!" leaked;
  p "capture windows pass every non-log line through to stdout"
    (List.equal String.equal (String.split_lines passed)
       (List.init windows ~f:(fun k -> line (k + 1))));
  gated ~aggregation:`Environment ~when_:lists_descriptors ~on:"a host without /dev/fd"
    "capture windows leave no descriptor open" (leaked = 0);
  let mk = Ll_test.node_factory ~first_id:1636000 ~dims:[| n |] () in
  let a = mk "rlsf_a" and x = mk "rlsf_x" in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add (Ll_test.get a [| at |]) (Ll_test.c 1.)))
  in
  let name = "rlsf_shift" in
  let o = Ll_test.optimize ~materialized:[ a; x ] ~name llc in
  let computes = "the logged routine computes its values"
  and quiet = "the logged run writes nothing to stdout"
  and filed = "the logged run's lines reach its stream file" in
  if String.equal backend_name "metal" then (
    ignore (Ll_test.link ~name o : Context.t * Context.routine);
    List.iter [ computes; quiet; filed ] ~f:(skipped ~backend:backend_name))
  else
    let av = [| 3.; -1.; 4.; 1. |] in
    let (ctx, values), out =
      stdout_of (fun () ->
          let ctx, _routine = Ll_test.run_routine ~name o ~seed:[ (a, av); (x, Ll_test.blank n) ] in
          (ctx, Context.get_values ctx x))
    in
    p_all2 computes values (Array.map av ~f:(fun v -> v +. 1.)) ~f:Float.equal;
    if not (String.is_empty out) then
      Stdio.eprintf "routine_log_stream_files: the run wrote to stdout:\n%s\n%!" out;
    p quiet (String.is_empty out);
    (* The stream files are named after the stream that ran the routine:
       [<backend>-<ordinal>-<id>]. *)
    let dir = Utils.log_files_dir () in
    let stream_lines =
      Stdlib.Sys.readdir dir |> Array.to_list
      |> List.filter ~f:(fun file ->
          String.is_prefix file ~prefix:(Context.backend_name ctx ^ "-")
          && String.is_suffix file ~suffix:".log")
      |> List.concat_map ~f:(fun file ->
          Stdio.In_channel.read_lines (Stdlib.Filename.concat dir file))
    in
    p_exists filed stream_lines ~f:(String.is_substring ~substring:"rlsf_x")
