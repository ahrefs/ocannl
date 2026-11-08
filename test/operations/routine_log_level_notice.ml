(* gh-ocannl-2064: capture initialization too, before the child can restore settings again. No
   backend is needed: the notice diagnoses configuration, before any routine is compiled. *)
open Base
open Verdict.Claims

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal "--notice-child") then (
    if Array.exists Stdlib.Sys.argv ~f:(String.equal "--restore-again") then (
      Utils.restore_settings ();
      Utils.restore_settings ());
    let enabled = Utils.settings.debug_log_from_routines && Utils.settings.log_level > 1 in
    if not (Bool.equal (Utils.debug_log_from_routines ()) enabled) then Stdlib.exit 1)
  else
    List.iter [ false; true ] ~f:(fun requested ->
        List.iter [ 0; 1; 2 ] ~f:(fun level ->
            List.iter [ false; true ] ~f:(fun restore ->
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
                p (label ^ ": child succeeds") (Poly.equal status (Unix.WEXITED 0));
                p (label ^ ": stdout stays empty") (String.is_empty out);
                let notices =
                  String.substr_index_all err ~may_overlap:false
                    ~pattern:"OCANNL notice: debug_log_from_routines=true has no effect"
                  |> List.length
                in
                p
                  (label ^ ": stderr notice matches the logging gate, once per process")
                  ((notices = if requested && level <= 1 then 1 else 0)
                  && ((not requested) || level > 1
                     || String.is_substring err ~substring:"log_level > 1")))))
