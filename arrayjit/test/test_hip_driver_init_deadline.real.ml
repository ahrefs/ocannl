(* gh-ocannl-1446: HIP's one-time driver initialization ([hipInit]) runs under the device-init
   deadline wherever it is forced from -- including [Hip_backend.Impl.hip_to_code] called directly,
   which forces it ahead of its own guarded device discovery and which no
   [Backends.With_device_init_deadline] wraps (test_hip_compile_failure_options is such a caller).

   A wedged driver cannot be produced on demand, so [Utils.driver_init_test_stall] simulates one: it
   sleeps 5 s right before the driver call. The test re-runs its own executable as the child, with
   the deadline pinned to 0.5 s on the child's commandline, and reads how the child ended: by
   SIGALRM, with the deadline's diagnostic naming hip, before the stall ran out. With the driver
   call outside any guarded region, the stall runs out and the child compiles and exits. *)

open Base
open Verdict.Claims

let claim =
  "a stuck HIP driver initialization that a direct Impl.hip_to_code call forces dies by the \
   device-init deadline, with the diagnostic naming hip"

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let child () =
  Utils.driver_init_test_stall :=
    Some
      (fun () ->
        Stdio.printf "stall-began\n%!";
        Unix.sleepf 5.0);
  (match
     Hip_backend.Impl.hip_to_code ~name:"driver_init_deadline_probe"
       {|extern "C" __global__ void driver_init_deadline_probe() {}|}
   with
  | (_ : Hiprtc.compile_to_code_result) -> Stdio.printf "compiled\n%!"
  | exception exn -> Stdio.printf "raised %s\n%!" (Exn.to_string exn));
  Stdio.printf "survived\n%!"

let read_lines path =
  Exn.protect
    ~f:(fun () -> Stdio.In_channel.read_lines path)
    ~finally:(fun () -> Stdlib.Sys.remove path)

let parent () =
  let exe = Stdlib.Sys.executable_name in
  let out = Stdlib.Filename.temp_file "ocannl-hip-init-" ".out" in
  let err = Stdlib.Filename.temp_file "ocannl-hip-init-" ".err" in
  let open_w path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let fd_out = open_w out and fd_err = open_w err in
  let pid =
    Unix.create_process exe
      [| exe; "--ocannl_device_init_deadline=0.5"; "child" |]
      Unix.stdin fd_out fd_err
  in
  Unix.close fd_out;
  Unix.close fd_err;
  let _, status = Unix.waitpid [] pid in
  let out = read_lines out and err = read_lines err in
  List.iter (out @ err) ~f:(Stdio.eprintf "child: %s (not part of the golden)\n");
  let by_sigalrm = match status with Unix.WSIGNALED s -> s = Stdlib.Sys.sigalrm | _ -> false in
  let says line = List.mem out line ~equal:String.equal in
  let diagnostic =
    List.exists err
      ~f:(String.is_prefix ~prefix:(Utils.device_init_deadline_tag ^ " hip device initialization"))
  in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  gated
    ~when_:(Utils.device_init_deadline_supported ())
    ~on:"a platform without SIGALRM" ~aggregation:`Environment claim
    (by_sigalrm && says "stall-began" && (not (says "survived")) && diagnostic)

let () =
  match Stdlib.Sys.argv with
  | [| _; _; "child" |] -> child ()
  | _ when String.equal backend_name "hip" -> parent ()
  | _ -> skipped ~backend:backend_name claim
