open Base
open Verdict.Claims

let diagnostic = "the planted refusal"
let large = String.make (256 * 1024) 'x'

let test () =
  let root = Unix.realpath (Stdlib.Filename.temp_dir "fresh process cases " "") in
  Exn.protect
    ~finally:(fun () ->
      Array.iter (Stdlib.Sys.readdir root) ~f:(fun name ->
          Unix.unlink (Stdlib.Filename.concat root name));
      Unix.rmdir root)
    ~f:(fun () ->
      let run ?env ?stderr args = Fresh_process.run ~temp_dir:root ?env ?stderr args in
      let clean label = p_empty label ~over:[ root ] (Array.to_list (Stdlib.Sys.readdir root)) in
      let refused = run [ "--refuse" ] in
      p "a refusal requires its exit status and causal diagnostic"
        (Fresh_process.matches ~stream:`Stderr ~exit:1 ~contains:[ diagnostic ] refused);
      p "the same diagnostic at status 0 is not a refusal"
        (not (Fresh_process.matches ~exit:1 ~contains:[ diagnostic ] (run [ "--pass" ])));
      p "status 1 with another diagnostic is not the expected refusal"
        (not (Fresh_process.matches ~exit:1 ~contains:[ "different refusal" ] refused));
      p "diagnostics cannot match through the wrong stream"
        (not (Fresh_process.matches ~stream:`Stdout ~exit:1 ~contains:[ diagnostic ] refused));
      let split = run [ "--split" ] in
      p "a causal diagnostic cannot be assembled across output streams"
        (not (Fresh_process.matches ~exit:1 ~contains:[ diagnostic ] split));
      p "complete diagnostic fragments may come from either captured stream"
        (Fresh_process.matches ~exit:1 ~contains:[ "the planted "; "refusal" ] split);
      let status, stdout, stderr = run [ "--large" ] in
      p "large streams stay separate and do not fill an unread pipe"
        (Poly.equal status (Unix.WEXITED 0)
        && String.equal stdout large
        && String.equal stderr ("stderr:" ^ large));
      clean "capture files are removed after normal and faulted child statuses";
      Unix.putenv "FRESH_PROCESS_INHERITED" "inherited";
      p "without an environment the child inherits the parent's"
        (Poly.equal (run [ "--env" ]) (Unix.WEXITED 0, "probe= inherited=inherited", ""));
      p "a given environment replaces the inherited one"
        (Poly.equal
           (run ~env:[| "FRESH_PROCESS_PROBE=planted" |] [ "--env" ])
           (Unix.WEXITED 0, "probe=planted inherited=", ""));
      let discarded, leaked =
        Stderr_capture.capture ~temp_dir:root ~echo:false (fun () ->
            run ~stderr:`Discard [ "--split" ])
      in
      p "a discarded stderr keeps the child's status and stdout and reaches no stream"
        (Poly.equal discarded (Unix.WEXITED 1, "the planted ", "") && String.is_empty leaked);
      let sink = Stdlib.Filename.concat root "sink" in
      let fd = Unix.openfile sink [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
      let into = run ~stderr:(`Into fd) [ "--split" ] in
      (* Still the caller's: a write after the child's succeeds. *)
      let still_open = Unix.write_substring fd "!" 0 1 = 1 in
      Unix.close fd;
      let sunk = Stdio.In_channel.read_all sink in
      Unix.unlink sink;
      p "stderr into the caller's descriptor reaches it and leaves it open"
        (Poly.equal into (Unix.WEXITED 1, "the planted ", "")
        && still_open && String.equal sunk "refusal!");
      clean "discarded and redirected stderr leave no capture files";
      let before = Stdlib.Sys.getcwd () in
      (* Relative executable resolution happens before the child's working directory changes. *)
      let status, stdout, _ =
        Fresh_process.run
          ~exe:(Stdlib.Filename.basename Stdlib.Sys.executable_name)
          ~cwd:root ~temp_dir:root [ "--cwd" ]
      in
      p "a relative executable still runs from another directory"
        (Poly.equal status (Unix.WEXITED 0) && String.equal stdout root);
      p "the parent directory is restored after a successful spawn"
        (String.equal before (Stdlib.Sys.getcwd ()));
      let spawn_failed =
        try
          ignore
            (Fresh_process.run
               ~exe:(Stdlib.Filename.concat root "missing.exe")
               ~cwd:root ~temp_dir:root []
              : Fresh_process.t);
          false
        with Unix.Unix_error (Unix.ENOENT, _, _) -> true
      in
      p "spawn failure restores the parent directory"
        (spawn_failed && String.equal before (Stdlib.Sys.getcwd ()));
      clean "spawn failure removes both capture files";
      let cwd_failed =
        try
          ignore
            (Fresh_process.run ~cwd:(Stdlib.Filename.concat root "missing-dir") ~temp_dir:root []
              : Fresh_process.t);
          false
        with Unix.Unix_error (Unix.ENOENT, _, _) -> true
      in
      p "a bad child directory preserves the parent directory"
        (cwd_failed && String.equal before (Stdlib.Sys.getcwd ()));
      clean "a bad child directory removes both capture files";
      let many_report = run [ "--report" ] in
      let _, _, echoed = many_report in
      p "echoed child Verdict markers cannot become a parent's line-initial report"
        (Fresh_process.matches ~exit:0
           ~contains:[ "  child | STOPPED EARLY:"; "  child | FAIL:"; "  child | FAILED:" ]
           many_report);
      p_none "no echoed child marker starts a parent log line" (String.split echoed ~on:'\n')
        ~f:(fun line ->
          String.is_prefix line ~prefix:"STOPPED EARLY"
          || String.is_prefix line ~prefix:"FAIL:"
          || String.is_prefix line ~prefix:"FAILED:");
      clean "diagnostic reporting also leaves no capture files")

let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  Stdlib.set_binary_mode_out Stdlib.stderr true;
  match Array.to_list Stdlib.Sys.argv with
  | [ _ ] -> test ()
  | [ _; "--refuse" ] ->
      Stdio.eprintf "%s" diagnostic;
      Stdlib.exit 1
  | [ _; "--pass" ] -> Stdio.eprintf "%s" diagnostic
  | [ _; "--split" ] ->
      Stdio.printf "the planted ";
      Stdio.eprintf "refusal";
      Stdlib.exit 1
  | [ _; "--large" ] ->
      Stdio.printf "%s" large;
      Stdio.eprintf "stderr:%s" large
  | [ _; "--cwd" ] -> Stdio.printf "%s" (Stdlib.Sys.getcwd ())
  | [ _; "--env" ] ->
      let get name = Option.value (Stdlib.Sys.getenv_opt name) ~default:"" in
      Stdio.printf "probe=%s inherited=%s" (get "FRESH_PROCESS_PROBE")
        (get "FRESH_PROCESS_INHERITED")
  | [ _; "--report" ] ->
      Fresh_process.report ~label:"faulted capture"
        (Unix.WEXITED 2, "FAIL: child claim\nFAILED: child summary\n", "STOPPED EARLY: child\n")
  | _ -> Stdlib.exit 2
