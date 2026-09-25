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
external write_fd2 : string -> unit = "ocannl_test_write_fd2"

let foreign_marker = "FOREIGN-C-STDERR-LINE"
let control_marker = "C-WRITE-TO-FD-2"
let ocaml_marker = "OCAML-STDERR-LINE"

(* One label per arm, spelled once: the skip below reports the claim it did not evaluate under the
   same label the evaluated form prints, so a golden line never says which of the two happened. *)
let arm_on_claim = "a foreign C library's stderr escapes a capture of fd 2"
let arm_off_claim = "without the setting, a foreign C library's stderr is captured with fd 2"

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
        write_fd2 control_marker;
        eprintf "%s\n%!" ocaml_marker)
  in
  let captured_foreign = String.is_substring captured ~substring:foreign_marker in
  let captured_ocaml = String.is_substring captured ~substring:ocaml_marker in
  (* The control (see the stubs): every claim below is about what a capture of fd 2 caught from a
     C-level write, and on a host where redirecting fd 2 from OCaml does not reach C at all, both of
     them are vacuous -- the "on" arm would pass for the wrong reason and the "off" arm would fail
     for one. Established by observation rather than by an `#ifdef` mirroring the stub's, so a host
     nobody anticipated is skipped rather than asserted about. *)
  let capture_reaches_c = String.is_substring captured ~substring:control_marker in
  eprintf "control: a C write to fd 2 was caught by the capture: %b (not part of the golden)\n"
    capture_reaches_c;
  (* The arm is the SETTING the rule asked for, not the effect it had: the effect is what the claims
     below are about, and on a host whose C [stderr] is not an assignable lvalue (Windows, and any
     libc outside glibc/Darwin) the stub declines and the requested arm has nothing to show.
     Reporting the effect here instead would move this line on those hosts, and the golden is one
     file for all of them. Descriptive, and deliberately not a `%b` claim: which arm ran is not a
     fact that can fail -- the rule runs both, and each asserts what holds under it. *)
  let requested = Utils.get_global_flag ~default:true ~arg_name:"detach_c_stderr" in
  printf "arm: detach_c_stderr=%s\n" (if requested then "on" else "off");
  let claim = if requested then arm_on_claim else arm_off_claim in
  (* Vacuous rather than false, in both directions: on a host that cannot detach [stderr] the "on"
     arm has nothing to escape, and on one where the capture never reaches C neither arm is about
     anything. Gated by the host and not by a backend -- this executable links none -- so the skip
     aggregates as [`Environment]. The claim's own label is unchanged either way, so the golden is
     one file for every host and a skip is visible only on stderr -- which [gated] guarantees by
     construction rather than by both branches remembering to use the same dialect. *)
  let skipped_on =
    if not capture_reaches_c then Some "a host whose fd-2 capture does not reach C writes"
    else if requested && not Utils.c_stderr_detached then
      Some "a C runtime with no assignable stderr"
    else None
  in
  gated ~aggregation:`Environment ~when_:(Option.is_none skipped_on)
    ~on:(Option.value skipped_on ~default:"")
    ~detail:(fun () -> Printf.sprintf "the capture held %S" captured)
    claim
    (if requested then not captured_foreign else captured_foreign);
  (* Unconditional: detaching the C stream must leave OCANNL's own diagnostics exactly where they
     were, or every `.expected` golden that reports through stderr would have moved with it. *)
  p "OCaml's own stderr follows fd 2 either way" captured_ocaml
