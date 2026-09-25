(* gh-ocannl-1036: device streams are torn down at process exit, and a busy device cannot hold the
   process there.

   A HIP stream that is never destroyed makes the ROCm runtime print `Resource leak detected by
   SharedSignalPool, 77 Signals leaked.` when the process ends -- a pure-HIP probe leaks exactly 77
   whatever the workload when it skips `hipStreamDestroy`, and every OCANNL hip process did, because
   nothing destroyed the one stream per device at exit. The HIP backend now registers an [at_exit]
   teardown ([Utils.bounded_exit_teardown]); this test runs [exit_stream_teardown_child] and reads
   what the child's exit left on its stderr, which is the only place the runtime reports it.

   Each property has a negative control. On hip, the teardown leaves no leak report, and with the
   teardown off ([exit_stream_teardown_timeout=0]) the report comes back -- so a clean run cannot
   pass for a ROCr build that stopped reporting leaks, which is skipped as vacuous instead.

   The bound is the other property: a crashing or hung process must not be held at exit. On hip, a
   stream with seconds of work queued at exit holds the teardown for the bound and no longer (that
   stream then goes undestroyed, and reported). On every backend, a stand-in stream that never
   becomes idle, and whose teardown would hang for an hour, delays exit by the bound and no more, on
   a normal exit and on an uncaught exception alike, while an idle one is torn down and a timeout of
   0 tears nothing down. A hung GPU cannot be staged on demand, so the stand-in exercises the same
   helper through a real [at_exit] in a real process. *)

open Base
open Stdio
open Verdict.Claims

let child_exe = (Sys.get_argv ()).(1)
let bound = 0.5
let leak_substring = "Signals leaked"

(* Runs the child with [args], stdout and stderr each to their own file; the environment, and so
   OCANNL_BACKEND, is inherited. Returns the exit code (a signal counts as a failure code), the wall
   time and both outputs. *)
let run_child args =
  let out_path = Stdlib.Filename.temp_file "ocannl-exit-teardown-" ".out" in
  let err_path = Stdlib.Filename.temp_file "ocannl-exit-teardown-" ".err" in
  let open_w path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out_fd = open_w out_path and err_fd = open_w err_path in
  let started = Mtime_clock.counter () in
  let pid =
    Unix.create_process child_exe (Array.of_list (child_exe :: args)) Unix.stdin out_fd err_fd
  in
  Unix.close out_fd;
  Unix.close err_fd;
  let _, status = Unix.waitpid [] pid in
  let seconds = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
  let code = match status with Unix.WEXITED c -> c | WSIGNALED _ | WSTOPPED _ -> 255 in
  let read path =
    let s = In_channel.read_all path in
    (try Stdlib.Sys.remove path with _ -> ());
    s
  in
  (code, seconds, read out_path, read err_path)

(* The signals the runtime reports leaked, summed over its reports: `... N Signals leaked.` *)
let leaked_signals err =
  String.split_lines err
  |> List.sum
       (module Int)
       ~f:(fun line ->
         match String.substr_index line ~pattern:leak_substring with
         | None -> 0
         | Some i ->
             String.prefix line i |> String.rstrip |> String.split ~on:' ' |> List.last
             |> Option.bind ~f:Int.of_string_opt |> Option.value ~default:1)

let backend_of out =
  String.split_lines out
  |> List.find_map ~f:(String.chop_prefix ~prefix:"backend: ")
  |> Option.value ~default:"<none>"

let device_arms () =
  let on_code, _, on_out, on_err = run_child [ "device" ] in
  let off_code, _, off_out, off_err =
    run_child [ "device"; "--ocannl_exit_stream_teardown_timeout=0" ]
  in
  let on_leaks = leaked_signals on_err and off_leaks = leaked_signals off_err in
  eprintf
    "device children ran on %s / %s; %d signals reported leaked with the teardown, %d without (not \
     part of the golden)\n\
     %!"
    (backend_of on_out) (backend_of off_out) on_leaks off_leaks;
  p "both device children exit normally" (on_code = 0 && off_code = 0);
  let on_claim = "the exit teardown leaves no leaked-signal report"
  and off_claim = "without the teardown, the runtime reports leaked signals" in
  if not (String.equal (backend_of on_out) "hip" && String.equal (backend_of off_out) "hip") then (
    let backend = backend_of on_out in
    skipped ~backend on_claim;
    skipped ~backend off_claim)
  else if off_leaks = 0 then (
    skipped ~aggregation:`Environment ~backend:"a ROCr build that does not report leaked signals"
      on_claim;
    skipped ~aggregation:`Environment ~backend:"a ROCr build that does not report leaked signals"
      off_claim)
  else (
    p on_claim (on_leaks = 0);
    p off_claim (off_leaks > 0))

let bound_flag = Printf.sprintf "--ocannl_exit_stream_teardown_timeout=%g" bound

(* The hung case on the device itself, as far as it can be staged: seconds of work queued and never
   synced. The backend's teardown must give up at the bound instead of draining the queue. What the
   driver does with the undestroyed stream after that is the driver's, so the claim is on the
   teardown's own duration, which the child times from an [at_exit] handler that runs after it. *)
let busy_device_arm () =
  let code, seconds, out, err = run_child [ "device_busy"; bound_flag ] in
  let backend = backend_of out in
  let teardown_seconds =
    String.split_lines out
    |> List.find_map ~f:(String.chop_prefix ~prefix:"exit teardown seconds: ")
    |> Option.bind ~f:(fun s -> Float.of_string_opt (String.strip s))
  in
  eprintf
    "device_busy on %s: exit %d after %.3fs, of which the exit teardown %s (not part of the golden)\n\
     %!"
    backend code seconds
    (Option.value_map teardown_seconds ~default:"did not report" ~f:(Printf.sprintf "%.3fs"));
  String.split_lines err
  |> List.filter ~f:(fun l ->
      String.is_substring l ~substring:"queued" || String.is_substring l ~substring:"still busy")
  |> List.iter ~f:(eprintf "device_busy child: %s\n");
  let claim = "a device still busy at exit delays the teardown by the bound and no more" in
  if not (String.equal backend "hip") then skipped ~backend claim
  else
    p claim
      (code = 0
      && String.is_substring err ~substring:"still busy"
      && Option.exists teardown_seconds ~f:(fun t -> Float.(t >= bound && t < bound +. 5.)))

let stand_in_arms () =
  let busy mode ~expected_code ~label =
    let code, seconds, out, err = run_child [ mode; bound_flag ] in
    eprintf "%s: exit %d after %.3fs (not part of the golden)\n%!" mode code seconds;
    p (label ^ ": the exit code is the program's own") (code = expected_code);
    (* Two-sided: at least the bound, so the helper did wait for the stream; well under the hour the
       stand-in's teardown would take, so it did not wait past the bound. *)
    p
      (label ^ ": exit is delayed by the bound and no more")
      Float.(seconds >= bound && seconds < bound +. 30.);
    p
      (label ^ ": the busy stream's teardown is never called")
      (not (String.is_substring out ~substring:Exit_stream_teardown_marker.teardown));
    p
      (label ^ ": the outcome is still_busy, reported on stderr")
      (String.is_substring out ~substring:"outcome: still_busy"
      && String.is_substring err ~substring:"still busy");
    err
  in
  ignore (busy "never_idle" ~expected_code:0 ~label:"never-idle stream, normal exit" : string);
  let err =
    busy "never_idle_raise" ~expected_code:2 ~label:"never-idle stream, uncaught exception"
  in
  p "the uncaught exception is still reported after the teardown gave up"
    (String.is_substring err ~substring:"deliberate uncaught exception");
  let code, _, out, _ = run_child [ "idle"; bound_flag ] in
  p "an idle stream is torn down at exit"
    (code = 0
    && String.is_substring out ~substring:Exit_stream_teardown_marker.teardown
    && String.is_substring out ~substring:"outcome: torn_down");
  let code, _, out, _ = run_child [ "idle"; "--ocannl_exit_stream_teardown_timeout=0" ] in
  p "a timeout of 0 turns the teardown off"
    (code = 0
    && (not (String.is_substring out ~substring:Exit_stream_teardown_marker.teardown))
    && String.is_substring out ~substring:"outcome: disabled")

let () =
  device_arms ();
  busy_device_arm ();
  stand_in_arms ()
