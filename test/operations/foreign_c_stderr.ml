(* gh-ocannl-1031: where a foreign C library's stderr goes when something takes over fd 2.

   The ROCm stack on a native Ubuntu 26.04 box prints `Signal 0x... time stamps may be invalid.`
   once per `hipEventRecord` — an assertions-enabled libhsa-runtime64 7.1.0 leaves ROCr's internal
   `debug_print` live, and ROCclr profiles every event marker whatever `hipEventDisableTiming` says,
   so ROCr reads back timestamps the hardware never wrote for a barrier packet. That is chatter, it
   belongs on stderr, and OCANNL's contract already keeps stdout as the data channel. ppx_expect
   breaks the contract: its capture redirects fd 1 AND fd 2 into the file it diffs against the
   `%expect` block (`ppx_expect_runtime_before_test` dup2s the temp file onto both), so six hip
   expect tests went red on 47,700 lines of driver chatter.

   [Utils.c_stderr_detached] answers it by giving the C runtime's [stderr] its own descriptor,
   duplicated from the real stderr at startup. This test performs exactly the redirection ppx_expect
   performs — dup2 a temp file onto fd 2 — and reports which writes it captured. Both arms are
   pinned by the rule next to it: with the setting on, the foreign line escapes the capture and
   OCaml's own stderr does not; with it off, the foreign line is captured again. The second arm is
   the negative control, so a golden that stayed green for having stopped exercising anything is not
   one of the readings this can produce.

   Nothing is filtered anywhere: in the detached arm the foreign line is on the process's real
   stderr, which is why the run log still shows it. *)

open Base
open Stdio
open Verdict.Claims

external write_c_stderr : string -> unit = "ocannl_test_write_c_stderr"

let foreign_marker = "FOREIGN-C-STDERR-LINE"
let ocaml_marker = "OCAML-STDERR-LINE"

(* The capture, written out rather than borrowed from a library, because it is the mechanism under
   test: the temp file becomes fd 2, both kinds of write happen, fd 2 is restored. *)
let capture_fd2 f =
  let path = Stdlib.Filename.temp_file "ocannl-fd2-" ".txt" in
  let captured_fd = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let saved_fd2 = Unix.dup Unix.stderr in
  Stdlib.flush Stdlib.stderr;
  Unix.dup2 captured_fd Unix.stderr;
  Exn.protect ~f ~finally:(fun () ->
      Stdlib.flush Stdlib.stderr;
      Unix.dup2 saved_fd2 Unix.stderr;
      Unix.close saved_fd2;
      Unix.close captured_fd);
  let contents = In_channel.read_all path in
  (try Stdlib.Sys.remove path with _ -> ());
  contents

let () =
  let captured =
    capture_fd2 (fun () ->
        write_c_stderr foreign_marker;
        eprintf "%s\n%!" ocaml_marker)
  in
  let captured_foreign = String.is_substring captured ~substring:foreign_marker in
  let captured_ocaml = String.is_substring captured ~substring:ocaml_marker in
  (* Descriptive, and deliberately not a `%b` claim: which arm ran is not a fact that can fail --
     the rule below runs both, and each arm asserts what holds under it. *)
  printf "arm: detach_c_stderr=%s\n" (if Utils.c_stderr_detached then "on" else "off");
  if Utils.c_stderr_detached then
    p "a foreign C library's stderr escapes a capture of fd 2" (not captured_foreign)
  else p "without the setting, a foreign C library's stderr is captured with fd 2" captured_foreign;
  (* Unconditional: detaching the C stream must leave OCANNL's own diagnostics exactly where they
     were, or every `.expected` golden that reports through stderr would have moved with it. *)
  p "OCaml's own stderr follows fd 2 either way" captured_ocaml
