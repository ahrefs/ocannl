(* The Verdict-only process behind sweep_harness.sh: the real skip emission, record files included.

   [replay [ACTION]] is the fake opam's stand-in for a test action (gh-ocannl-1114): it re-announces
   through [Verdict.skipped] every skip record of the fixture text on stdin, so the per-action
   record file is written by Verdict itself rather than restated by the harness. Its arguments are
   the action's identity -- a rerun replays under the same ones and so rewrites the same file -- and
   the claims come on stdin precisely so that a retry announcing different skips is still the same
   action. *)

let replay () =
  let prefix = "OCANNL_TOOL_VERDICT_SKIP\t" in
  let plen = String.length prefix in
  let rec loop () =
    match In_channel.input_line stdin with
    | None -> ()
    | Some line ->
        (if String.length line > plen && String.sub line 0 plen = prefix then
           match String.split_on_char '\t' (String.sub line plen (String.length line - plen)) with
           | [ scope; _exe; claim ] ->
               let aggregation =
                 match scope with
                 | "backend" -> `Backend
                 | "environment" -> `Environment
                 | "outside-sweep" -> `Outside_sweep
                 | _ -> failwith ("unknown skip scope in fixture: " ^ scope)
               in
               Verdict.skipped ~aggregation ~backend:"fixture" claim
           | _ -> failwith ("malformed skip record in fixture: " ^ line));
        loop ()
  in
  loop ()

let () =
  if Array.length Sys.argv >= 2 && Sys.argv.(1) = "replay" then (
    if Array.length Sys.argv > 3 then (
      prerr_endline "usage: verdict_skip_probe replay [ACTION]";
      exit 2);
    replay ();
    exit 0);
  if Array.length Sys.argv < 2 || Array.length Sys.argv > 3 then (
    prerr_endline
      "usage: verdict_skip_probe BACKEND [execute-environment|environment-as-backend|execute-all]";
    exit 2);
  let mode = if Array.length Sys.argv = 3 then Some Sys.argv.(2) else None in
  if mode = Some "execute-all" then (
    Verdict.p "common unevaluated claim" true;
    Verdict.p "common environment-gated claim" true)
  else (
    Verdict.skipped ~backend:Sys.argv.(1) "common unevaluated claim";
    match mode with
    | Some "execute-environment" -> Verdict.p "common environment-gated claim" true
    | Some "environment-as-backend" ->
        Verdict.skipped ~backend:Sys.argv.(1) "common environment-gated claim"
    | Some mode -> failwith ("unknown verdict probe mode: " ^ mode)
    | None ->
        Verdict.skipped ~aggregation:`Environment ~backend:"fixture gate"
          "common environment-gated claim")
