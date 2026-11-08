(* gh-ocannl-2064: capture initialization too, before the child can restore settings again. No
   backend is needed: the notice diagnoses configuration, before any routine is compiled. *)
open Base
open Verdict.Claims

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal "--notice-child") then (
    if Array.exists Stdlib.Sys.argv ~f:(String.equal "--restore-again") then (
      Utils.restore_settings ();
      Utils.restore_settings ()))
  else
    List.iter
      [ (true, 0, false); (true, 1, false); (true, 1, true); (true, 2, false); (false, 0, false) ]
      ~f:(fun (requested, level, restore) ->
        let label =
          Printf.sprintf "requested=%s level=%d restore=%s" (Bool.to_string requested) level
            (Bool.to_string restore)
        in
        let args =
          [
            "--notice-child";
            "--ocannl_no_config_file=true";
            "--ocannl_log_config_sourcing=false";
            "--ocannl_log_main_domain_to_stdout=false";
            "--ocannl_debug_log_from_routines=" ^ Bool.to_string requested;
            Printf.sprintf "--ocannl_log_level=%d" level;
          ]
          @ if restore then [ "--restore-again" ] else []
        in
        let ((status, out, err) as run) = Fresh_process.run args in
        Fresh_process.report ~label run;
        let notices =
          String.substr_index_all err ~may_overlap:false ~pattern:Utils.routine_log_level_notice
          |> List.length
        in
        p
          (label ^ ": succeeds with empty stdout and the expected stderr notice count")
          (Poly.equal status (Unix.WEXITED 0)
          && String.is_empty out
          && notices = if requested && level <= 1 then 1 else 0))
