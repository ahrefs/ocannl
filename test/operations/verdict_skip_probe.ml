(* The Verdict-only process behind sweep_harness.sh: the real skip emission, record files included.

   [replay [ACTION]] is the fake opam's stand-in for a test action (gh-ocannl-1114): it re-announces
   through [Verdict.skipped] every skip record of the fixture text on stdin, and through
   [Verdict.p_unless] every hatched claim's BYPASS or CHECKED record (gh-ocannl-996), so the
   per-action record file is written by Verdict itself rather than restated by the harness. Its
   arguments are the action's identity -- a rerun replays under the same ones and so rewrites the
   same file -- and the claims come on stdin precisely so that a retry announcing different skips is
   still the same action.

   [BACKEND relaunch] skips a claim, then relaunches this executable with the SAME arguments in the
   same directory -- the record identity of its own action, the `cc_march_census` shape -- marked as
   the child only by an environment variable no record key reads. The child records nothing of its
   own; were it handed the records directory, its start would truncate the parent's file and erase
   the skip the parent already flushed. *)

let child_marker = "VERDICT_SKIP_PROBE_RELAUNCHED"

let relaunch backend =
  Verdict.skipped ~backend "common unevaluated claim";
  let exe = Sys.executable_name in
  let env = Array.append [| child_marker ^ "=1" |] (Unix.environment ()) in
  let pid = Unix.create_process_env exe Sys.argv env Unix.stdin Unix.stdout Unix.stderr in
  let _, status = Unix.waitpid [] pid in
  Verdict.p "the relaunched child exits normally" (status = Unix.WEXITED 0)

let replay () =
  let prefix = "OCANNL_TOOL_VERDICT_SKIP\t" in
  let plen = String.length prefix in
  let hatched kind =
    let prefix = "OCANNL_TOOL_VERDICT_" ^ kind ^ "\t" in
    fun line ->
      let n = String.length prefix in
      if String.length line > n && String.sub line 0 n = prefix then
        match String.split_on_char '\t' (String.sub line n (String.length line - n)) with
        | [ hatch; _exe; claim ] -> Some (hatch, claim)
        | _ -> failwith ("malformed hatched-claim record in fixture: " ^ line)
      else None
  in
  let bypass = hatched "BYPASS" and checked = hatched "CHECKED" in
  let rec loop () =
    match In_channel.input_line stdin with
    | None -> ()
    | Some line ->
        (match (bypass line, checked line) with
        | Some (hatch, claim), _ -> Verdict.p_unless ~unless:[ (hatch, true) ] claim true
        | None, Some (hatch, claim) -> Verdict.p_unless ~unless:[ (hatch, false) ] claim true
        | None, None -> ());
        (if String.length line > plen && String.sub line 0 plen = prefix then
           let replay aggregation claim = Verdict.skipped ~aggregation ~backend:"fixture" claim in
           match String.split_on_char '\t' (String.sub line plen (String.length line - plen)) with
           | [ "backend"; _exe; claim ] -> replay `Backend claim
           | [ "environment"; _exe; claim ] -> replay `Environment claim
           | [ "outside-sweep"; _exe; claim; owner ] -> (
               (* The fixture's owner, not the tree's: read back through Verdict's own grammar so
                  the replayed record carries it verbatim. *)
               match Verdict.sweep_owner_of_field owner with
               | Some owner -> replay (`Outside_sweep owner) claim
               | None -> failwith ("malformed outside-sweep owner in fixture: " ^ line))
           | _ -> failwith ("malformed skip record in fixture: " ^ line));
        loop ()
  in
  loop ()

let () =
  if Sys.getenv_opt child_marker = Some "1" then exit 0;
  if Array.length Sys.argv = 3 && Sys.argv.(2) = "relaunch" then (
    relaunch Sys.argv.(1);
    exit 0);
  if Array.length Sys.argv >= 2 && Sys.argv.(1) = "replay" then (
    if Array.length Sys.argv > 3 then (
      prerr_endline "usage: verdict_skip_probe replay [ACTION]";
      exit 2);
    replay ();
    exit 0);
  if Array.length Sys.argv < 2 || Array.length Sys.argv > 3 then (
    prerr_endline
      "usage: verdict_skip_probe BACKEND \
       [execute-environment|environment-as-backend|execute-all|relaunch]";
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
