(* gh-ocannl-1370: the ROCm runtime's per-event `Signal 0x... time stamps may be invalid.` is
   collapsed into a count on the C runtime's stderr, and nothing else it carries is touched.

   An assertions-enabled ROCr (Ubuntu 26.04's libhsa-runtime64 7.1.0, on an APU) prints that line
   once per recorded event -- 268,172 for one autotuning test. gh-ocannl-1031 already keeps it out
   of ppx_expect goldens, but it still filled the run log, and Dune truncates a long action output
   from the middle, so a HIP test's own SKIPPED lines and diagnostics were what went missing. The
   HIP backend now forces [Utils.c_stderr_noise_collapsed] when it opens a device, which replaces
   the C stream by one that writes the first such line, a note, and at exit a count.

   The line is only ever printed at exit or by a foreign library, so it is observed from outside:
   this parent runs [foreign_c_stderr_noise_child] -- which installs the stream the way the HIP
   backend does and makes the scripted writes of [Foreign_c_stderr_noise_script] -- and reads back
   the child's stderr. Both arms of the setting run, pinned on the child's commandline, which no
   environment variable outranks: with it off, the stderr is every write verbatim, the negative
   control that the scripted lines are recognizable at all and that the "on" claims are not
   satisfied by a child that wrote nothing. *)

open Base
open Stdio
open Verdict.Claims
module S = Foreign_c_stderr_noise_script

let child_exe = (Sys.get_argv ()).(1)

(* What the child wrote from the begin marker on, with Windows' text-mode CRs dropped: whatever the
   configuration's startup wrote before the marker is not the subject. *)
let scripted stderr_text =
  let text = String.filter stderr_text ~f:(fun c -> not (Char.equal c '\r')) in
  match String.substr_index text ~pattern:S.begin_marker with
  | Some i -> String.drop_prefix text i
  | None -> ""

let run_child ?(close = false) setting =
  let ((status, stdout_text, stderr_text) as child) =
    Fresh_process.run ~exe:child_exe
      ([ "--ocannl_collapse_foreign_stderr_noise=" ^ Bool.to_string setting ]
      @ if close then [ "close" ] else [])
  in
  (match status with
  | Unix.WEXITED 0 -> ()
  | _ -> Fresh_process.report ~label:("child, collapsing " ^ Bool.to_string setting) child);
  let installed = String.is_substring stdout_text ~substring:"stream: collapsing" in
  let text = scripted stderr_text in
  eprintf "child's stderr with collapsing %s%s, from the marker (not part of the golden):\n%s\n%!"
    (if setting then "on" else "off")
    (if close then ", then fclose(stderr)" else "")
    (Fresh_process.prefixed (String.rstrip text));
  (status, installed, text)

let lines text = String.split_lines text
let ocannl_line l = String.is_prefix l ~prefix:"OCANNL: "

let () =
  let status_on, installed_on, on = run_child true in
  let status_off, installed_off, off = run_child false in
  let status_closed, _, closed = run_child ~close:true true in
  let exited_cleanly = function Unix.WEXITED 0 -> true | _ -> false in
  p "both children exit normally" (exited_cleanly status_on && exited_cleanly status_off);
  (* The negative control: every scripted write reaches stderr unchanged when the setting is off,
     and the stream was not even installed. *)
  p "with collapsing off, the C stream is left as it was" (not installed_off);
  p "with collapsing off, every write reaches stderr verbatim and in order"
    (String.equal off (String.concat S.writes));
  p "with collapsing off, every recognized line is printed"
    (List.count (lines off) ~f:(fun l ->
         String.is_prefix l ~prefix:"Signal 0x7"
         && String.is_suffix l ~suffix:" time stamps may be invalid.")
    = S.recognized_count);
  (* Gated on the predicate the stubs are compiled under, not on the stub's report of success: off
     glibc and Darwin the stub declines and the "on" arm has nothing to show, but a stub that
     stopped installing on a host that can must fail here rather than turn every claim below into a
     skip. Gated by the host, not a backend -- nothing here links one. *)
  let when_ = Utils.c_stderr_replaceable ()
  and on_ = "a C runtime whose stderr cannot be replaced" in
  gated ~aggregation:`Environment ~when_ ~on:on_ "with collapsing on, the C stream is replaced"
    installed_on;
  gated ~aggregation:`Environment ~when_ ~on:on_
    ~detail:(fun () -> Printf.sprintf "the child's stderr was %S" on)
    "with collapsing on, every other write reaches stderr verbatim and in order, and only the \
     first recognized line survives"
    (List.equal String.equal
       (List.filter (lines on) ~f:(fun l -> not (ocannl_line l)))
       S.collapsed_lines);
  let notes = List.filter (lines on) ~f:ocannl_line in
  let first_line = S.recognized "7f3a1c000100" |> String.rstrip in
  let note_follows_first =
    let rec go = function
      | a :: b :: rest -> (String.equal a first_line && ocannl_line b) || go (b :: rest)
      | _ -> false
    in
    go (lines on)
  in
  gated ~aggregation:`Environment ~when_ ~on:on_
    "with collapsing on, a note follows the first recognized line" note_follows_first;
  let summary = Printf.sprintf "collapsed %d further" (S.recognized_count - 1) in
  gated ~aggregation:`Environment ~when_ ~on:on_
    "with collapsing on, the exit summary is the last line and counts every later recognized line"
    (List.length notes = 2
    && Option.value_map
         (List.last (lines on))
         ~default:false
         ~f:(fun l -> ocannl_line l && String.is_substring l ~substring:summary));
  (* An [fclose(stderr)] after the writes: the close callback reports the count and retires the exit
     hook, so the child exits normally with exactly what it would have written at exit. *)
  gated ~aggregation:`Environment ~when_ ~on:on_
    ~detail:(fun () -> Printf.sprintf "the child's stderr was %S" closed)
    "with collapsing on, closing the C stream reports the count and the process still exits cleanly"
    (exited_cleanly status_closed && String.equal closed on)
