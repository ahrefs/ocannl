(* gh-ocannl-1296: the shared stderr capture and progress-line reader of
   test/support/stderr_capture.

   The capture's contract is what happens on the way OUT: fd 2 restored, the captured text written
   back to it, the temporary file removed, the descriptors closed, and an exception from inside
   re-raised only after all of that. Four of the five copies this module replaced kept none of it
   when the captured function raised -- fd 2 stayed on a file the run then lost, so the exception's
   own report, and Verdict's teardown after it, went nowhere. So the raising path is the one checked
   hardest here: in process, nested inside a second capture that sees what the first wrote back, and
   in a child whose uncaught exception must reach its real stderr AFTER the captured text.

   The reader's contract is the line format [arrayjit/lib/autotune.mli] documents: [wall_s] and
   [event] first, bare words or OCaml-quoted values, and [None] for anything else. *)

open Base
open Verdict.Claims
module Progress = Stderr_capture.Progress

let marker = "STDERR-CAPTURE-MARKER"

(* The identity of whatever fd 2 is: restored means this is what it was before. *)
let fd2_identity () =
  let st = Unix.fstat Unix.stderr in
  (st.Unix.st_dev, st.Unix.st_ino)

let fresh_dir () =
  let dir = Stdlib.Filename.temp_file "stderr_capture_cases" ".d" in
  Stdlib.Sys.remove dir;
  Unix.mkdir dir 0o700;
  dir

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: "raise-uncaught" :: _ ->
      (* The child: an exception escapes the capture, and the process. *)
      ignore
        (Stderr_capture.capture (fun () ->
             Stdio.eprintf "%s\n%!" marker;
             failwith "planted inside the capture")
          : unit * string)
  | _ ->
      (* --- The progress-line reader. --- *)
      let parsed =
        Progress.parse
          "autotune-progress: wall_s=1.5 event=arm_start arm=\"flip 2/3 \\\"q\\\" x\" flip=2/3"
      in
      p "a progress line reads its leading wall_s and event"
        (match parsed with
        | Some { event = "arm_start"; wall_s; _ } -> Float.equal wall_s 1.5
        | _ -> false);
      p "a quoted value keeps its spaces and is unescaped as an OCaml string"
        (Option.equal String.equal
           (Option.bind parsed ~f:(fun l -> Progress.field l "arm"))
           (Some "flip 2/3 \"q\" x"));
      p "a bare value ends at the next space"
        (Option.equal String.equal
           (Option.bind parsed ~f:(fun l -> Progress.field l "flip"))
           (Some "2/3"));
      p_none "a line breaking the documented shape reads as none"
        [
          "autotune-progress: event=arm_start wall_s=1.5";
          "autotune-progress: wall_s=soon event=arm_start";
          "autotune-progress: wall_s=1.5 event=arm_start arm=\"never closed";
          "autotune-progress: wall_s=1.5 event=arm_start arm=\"x\"tail";
          "autotune-progress: wall_s=1.5 event=arm_start empty=";
          "autotune-progress: wall_s=1.5 event=arm_start bare";
          "autotune-log: wall_s=1.5 event=arm_start";
        ] ~f:(fun line -> Option.is_some (Progress.parse line));
      let lines =
        [
          "autotune-progress: wall_s=0.1 event=stage stage=untuned_control";
          "unrelated chatter";
          "autotune-progress: wall_s=0.2 event=search_done outcome=searched";
          "autotune-progress: wall_s=0.3 event=stage stage=base_compile";
        ]
      in
      p "events and count select the progress lines by their fields"
        (List.length (Progress.events lines "stage") = 2
        && Progress.count lines [ ("event", "stage"); ("stage", "untuned_control") ] = 1
        && Progress.count lines [ ("event", "search_done") ] = 1);
      (* --- The capture, returning. --- *)
      let before = fd2_identity () in
      let result, text =
        Stderr_capture.capture ~echo:false (fun () ->
            Stdio.eprintf "%s\n%!" marker;
            42)
      in
      p "a capture answers the function's result and the text it wrote to stderr"
        (result = 42 && String.equal text (marker ^ "\n"));
      p "a capture that returned leaves fd 2 what it was" (Poly.equal before (fd2_identity ()));
      (* --- The capture, raising: nested in a second capture, which sees what the first wrote back
         once it had restored fd 2 (to the outer capture's file). --- *)
      let dir = fresh_dir () in
      let (raised, restored), outer =
        Stderr_capture.capture ~echo:false (fun () ->
            (* fd 2 is the outer capture's file here, and must be again once the inner one
               raised. *)
            let outer_fd2 = fd2_identity () in
            let raised =
              match
                Stderr_capture.capture ~temp_dir:dir (fun () ->
                    Stdio.eprintf "%s\n%!" marker;
                    failwith "planted inside the capture")
              with
              | _ -> None
              | exception Failure message -> Some message
            in
            (raised, Poly.equal outer_fd2 (fd2_identity ())))
      in
      p "an exception inside a capture reaches its caller unchanged"
        (Option.equal String.equal raised (Some "planted inside the capture"));
      p "a capture that raised restored fd 2 before re-raising" restored;
      p "a capture that raised wrote its captured text back to the restored stderr"
        (String.equal outer (marker ^ "\n"));
      let left = Stdlib.Sys.readdir dir in
      p_empty "a capture that raised removed its temporary file" ~over:[ dir ] (Array.to_list left);
      p "a capture that raised leaves fd 2 what it was" (Poly.equal before (fd2_identity ()));
      Array.iter left ~f:(fun file -> Stdlib.Sys.remove (Stdlib.Filename.concat dir file));
      Unix.rmdir dir;
      (* --- The capture, failing to set up: no file to route into. --- *)
      let missing = Stdlib.Filename.concat dir "absent" in
      p "a capture whose temporary file cannot be created raises without moving fd 2"
        (match Stderr_capture.capture ~temp_dir:missing (fun () -> ()) with
        | _ -> false
        | exception Sys_error _ -> Poly.equal before (fd2_identity ()));
      (* --- An exception escaping the process from inside a capture: its report must follow the
         captured text on the process's real stderr. --- *)
      let status, _, child_stderr = Fresh_process.run [ "raise-uncaught" ] in
      let index sub = String.substr_index child_stderr ~pattern:sub in
      let reported =
        match (index marker, index "planted inside the capture") with
        | Some m, Some e -> m < e
        | _ -> false
      in
      if not reported then Fresh_process.report ~label:"raise-uncaught" (status, "", child_stderr);
      p
        "an exception escaping a capture is reported on the real stderr, after the text the \
         capture caught"
        (Poly.equal status (Unix.WEXITED 2) && reported)
