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

(* Longer than any child here should take by far -- the slowest, on hip, compiles one matmul and
   waits out a 0.5s bound -- yet short of a suite's own job timeout, so a child that hangs at exit
   (the regression this test exists to catch) fails the claims instead of stalling the suite. *)
let child_deadline = 120.

(* Runs the child with [args], stdout and stderr each to their own file; the environment, and so
   OCANNL_BACKEND, is inherited. Returns the exit code (a signal, or being killed at the deadline,
   counts as 255), the wall time and both outputs. *)
let run_child args =
  let out_path = Stdlib.Filename.temp_file "ocannl-exit-teardown-" ".out" in
  let err_path = Stdlib.Filename.temp_file "ocannl-exit-teardown-" ".err" in
  let open_w path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out_fd = open_w out_path and err_fd = open_w err_path in
  let started = Mtime_clock.counter () in
  let elapsed () = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
  let pid =
    Unix.create_process child_exe (Array.of_list (child_exe :: args)) Unix.stdin out_fd err_fd
  in
  Unix.close out_fd;
  Unix.close err_fd;
  let rec reap () =
    match Unix.waitpid [ Unix.WNOHANG ] pid with
    | 0, _ when Float.(elapsed () < child_deadline) ->
        Unix.sleepf 0.005;
        reap ()
    | 0, _ ->
        eprintf "child %s still running after %gs: killed (not part of the golden)\n%!"
          (String.concat ~sep:" " args) child_deadline;
        (try Unix.kill pid Stdlib.Sys.sigkill with Unix.Unix_error _ -> ());
        ignore (Unix.waitpid [] pid : int * Unix.process_status);
        255
    | _, Unix.WEXITED c -> c
    | _, (WSIGNALED _ | WSTOPPED _) -> 255
  in
  let code = reap () in
  let seconds = elapsed () in
  let read path =
    let s = In_channel.read_all path in
    (try Stdlib.Sys.remove path with _ -> ());
    s
  in
  (code, seconds, read out_path, read err_path)

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
  (* Both arms pin the setting on the command line, which outranks an ambient one. *)
  let on_code, _, on_out, on_err =
    run_child
      [
        "device";
        Printf.sprintf "--ocannl_exit_stream_teardown_timeout=%g"
          Utils.default_exit_stream_teardown_timeout;
      ]
  in
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
  (* On every backend: the program's own [at_exit], registered before the device opened, still reads
     the device (7 * 2 + 1), so the backend's teardown ran after it rather than under it. *)
  p "an at_exit handler registered before the device opened still reads it at exit"
    (String.is_substring on_out ~substring:"read at exit: 15\n");
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
  (* The teardown's own reading of how long it waited: nothing the child registers can run after the
     backend's handler, which the backend registers at its module initialization. *)
  let teardown_seconds =
    String.split_lines err
    |> List.find_map ~f:(fun line ->
        match String.substr_index line ~pattern:"still busy after " with
        | None -> None
        | Some i ->
            String.drop_prefix line (i + String.length "still busy after ")
            |> String.lsplit2 ~on:'s' |> Option.map ~f:fst
            |> Option.bind ~f:Float.of_string_opt)
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
  (* An [inf] bound would poll forever; it falls back to the default instead. Two-sided against that
     default, which is what [Utils.bounded_exit_teardown] reads when the key is unset. *)
  let default_bound = Utils.default_exit_stream_teardown_timeout in
  let code, seconds, out, err =
    run_child [ "never_idle"; "--ocannl_exit_stream_teardown_timeout=inf" ]
  in
  eprintf "never_idle with an infinite bound: exit %d after %.3fs (not part of the golden)\n%!" code
    seconds;
  p "a non-finite bound falls back to the default, reported on stderr"
    (code = 0
    && Float.(seconds >= default_bound && seconds < default_bound +. 30.)
    && String.is_substring out ~substring:"outcome: still_busy"
    && String.is_substring err ~substring:"not a finite number");
  (* One bound for every resource, not one each: two never-idle stand-ins under a 1s bound exit in
     under 2s, which a bound restarted per resource could not. *)
  let code, seconds, out, _ =
    run_child [ "two_never_idle"; "--ocannl_exit_stream_teardown_timeout=1" ]
  in
  eprintf "two_never_idle: exit %d after %.3fs (not part of the golden)\n%!" code seconds;
  p "two busy streams share one bound"
    (code = 0
    && Float.(seconds >= 1. && seconds < 2.)
    && List.count (String.split_lines out) ~f:(String.equal "outcome: still_busy") = 2);
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
