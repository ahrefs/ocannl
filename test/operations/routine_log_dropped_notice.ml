(* gh-ocannl-1999: a logged routine whose lines nothing reads says so on stderr, once.

   With [debug_log_to_stream_files] off, a logged routine's lines go to the debug trace
   ([Utils.log_trace_tree]), which a build without [OCANNL_LOG_LEVEL_UTILS] compiles out: every line
   was dropped, and nothing told the user who had turned routine logging on to read them. The
   kernels are compiled for logging all the same (serial, unvectorized, formatting each line), and
   several tests rely on that as a code generation switch, so the lines are still produced; what
   changed is that the first batch of them that reaches the no-op prints
   [Utils.routine_logs_dropped_notice].

   The rule runs this with routine logging on and the stream-file key off. The C backends hand each
   routine's log file to that sink after the run and Metal drains its buffered shader logs to it at
   synchronization; CUDA and HIP print device-side, and only a caller's [Utils.capture_stdout_logs]
   window hands their lines to it. So both runs here sit in such a window, which makes the one
   sequence hold on every backend; the backend that ran is printed on stderr, since the golden is
   the same everywhere. A build that sets [OCANNL_LOG_LEVEL_UTILS] delivers the lines to the trace
   and prints no notice: this test is for the default build. *)

open Base
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* [f]'s result with every byte written to file descriptors 1 and 2 while it ran, by OCaml and C
   code alike. Both stdio layers are flushed on each side, and the descriptors are restored on every
   path. *)
let outputs_of f =
  let temp () = Stdlib.Filename.temp_file "routine_log_dropped_notice" ".txt" in
  let out_file = temp () and err_file = temp () in
  let remove file = try Stdlib.Sys.remove file with Sys_error _ -> () in
  Exn.protect
    ~finally:(fun () -> List.iter [ out_file; err_file ] ~f:remove)
    ~f:(fun () ->
      let flush () =
        Stdio.Out_channel.flush Stdio.stdout;
        Stdio.Out_channel.flush Stdio.stderr;
        Utils.flush_c_streams ()
      in
      flush ();
      let saved_out = Unix.dup Unix.stdout and saved_err = Unix.dup Unix.stderr in
      let route file target =
        let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
        Exn.protect ~finally:(fun () -> Unix.close fd) ~f:(fun () -> Unix.dup2 fd target)
      in
      let result =
        Exn.protect
          ~finally:(fun () ->
            flush ();
            Unix.dup2 saved_out Unix.stdout;
            Unix.dup2 saved_err Unix.stderr;
            Unix.close saved_out;
            Unix.close saved_err)
          ~f:(fun () ->
            route out_file Unix.stdout;
            route err_file Unix.stderr;
            f ())
      in
      let err = Stdio.In_channel.read_all err_file in
      (* Whatever else the run said on stderr stays readable in the test's own. *)
      Stdio.eprintf "%s%!" err;
      (result, Stdio.In_channel.read_all out_file, err))

let n = 4

(* Runs a routine of [n] logged stores into a fresh node named [label] inside a capture window that
   also synchronizes, and gives the context, the values it stored and the ones it should have. *)
let run_logged ?ctx ~first_id ~label () =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk (label ^ "_src") and x = mk label in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add (Ll_test.get a [| at |]) (Ll_test.c 1.)))
  in
  let o = Ll_test.optimize ~materialized:[ a; x ] ~name:label llc in
  let av = Array.init n ~f:Float.of_int in
  Utils.capture_stdout_logs (fun () ->
      let ctx, _routine =
        Ll_test.run_routine ?ctx ~name:label o ~seed:[ (a, av); (x, Ll_test.blank n) ]
      in
      let got = Context.get_values ctx x in
      Context.sync ctx;
      (ctx, got, Array.map av ~f:(fun v -> v +. 1.)))

let notices err =
  List.length
    (String.substr_index_all err ~may_overlap:false ~pattern:Utils.routine_logs_dropped_notice)

let () =
  Stdio.eprintf "routine_log_dropped_notice: backend %s (not part of the golden)\n%!" backend_name;
  p "the run logs its routines, and not to stream files"
    (Utils.debug_log_from_routines ()
    && not (Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files"));
  p "the notice names the key that keeps the lines"
    (String.is_substring Utils.routine_logs_dropped_notice ~substring:"debug_log_to_stream_files");
  let (ctx, got, want), out, err = outputs_of (run_logged ~first_id:1999000 ~label:"rldn_first") in
  p_all2 "rldn_first: the logged routine computes its values" got want ~f:Float.equal;
  if not (String.is_empty out) then
    Stdio.eprintf "routine_log_dropped_notice: the first run wrote to stdout:\n%s\n%!" out;
  p "the first logged run writes nothing to stdout" (String.is_empty out);
  p "the first logged run says once on stderr that its lines are dropped" (notices err = 1);
  let (_, got, want), out, err =
    outputs_of (run_logged ~ctx ~first_id:1999100 ~label:"rldn_second")
  in
  p_all2 "rldn_second: the logged routine computes its values" got want ~f:Float.equal;
  p "the second logged run writes nothing to stdout" (String.is_empty out);
  p "the second logged run does not repeat the notice" (notices err = 0)
