open Base
open Verdict.Claims

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 1]

let probe_arg = "--flushing-location-format-probe"
let probe_message = "flushing location format probe"
let%diagn_sexp emit_probe () = [%log "flushing location format probe"]

type destination = File | Stdout

let destination_name = function File -> "file" | Stdout -> "stdout"

type child_result = {
  status : Unix.process_status;
  emitted : string;
  stdout_text : string;
  stderr_text : string;
}

(* Each configuration gets a fresh process because Utils snapshots command-line configuration when
   its debug runtime is first constructed. Capture both streams in files so neither pipe can fill
   while the parent is waiting on the other. *)
let run_child ~destination ~location_format =
  let stem =
    Printf.sprintf "flushing_location_format_%s_%s" (destination_name destination) location_format
  in
  let args =
    [
      probe_arg;
      "--ocannl_clean_up_log_files_on_startup=true";
      "--ocannl_debug_backend=flushing";
      "--ocannl_location_format=" ^ location_format;
      "--ocannl_log_file_stem=" ^ stem;
      "--ocannl_log_level=1";
      (match destination with
      | File -> "--ocannl_log_main_domain_to_stdout=false"
      | Stdout -> "--ocannl_log_main_domain_to_stdout=true");
      "--ocannl_time_tagged=not_tagged";
    ]
  in
  let status, stdout_text, stderr_text = Fresh_process.run args in
  let emitted =
    match destination with
    | Stdout -> stdout_text
    | File ->
        Option.try_with (fun () -> Stdio.In_channel.read_all (Utils.diagn_log_file (stem ^ ".log")))
        |> Option.value ~default:""
  in
  { status; emitted; stdout_text; stderr_text }

let status_ok = function Unix.WEXITED 0 -> true | _ -> false

let report_failure ~destination ~location_format result =
  Fresh_process.report
    ~label:
      (Printf.sprintf "%s flushing location_format=%s" (destination_name destination)
         location_format)
    (result.status, result.stdout_text, result.stderr_text);
  Stdio.eprintf "Emitted child log:\n%s\n" (Fresh_process.prefixed result.emitted)

let check_case ~destination ~location_format ~want_location =
  let result = run_child ~destination ~location_format in
  let has_probe = String.is_substring result.emitted ~substring:probe_message in
  let location_marker = "flushing_location_format.ml\":" in
  let has_location = String.is_substring result.emitted ~substring:location_marker in
  let ok = status_ok result.status && has_probe && Bool.equal has_location want_location in
  if not ok then report_failure ~destination ~location_format result;
  let behavior = if want_location then "emits source positions" else "omits source positions" in
  pf "%s flushing %s %s" (destination_name destination) location_format behavior ok

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal probe_arg) then emit_probe ()
  else
    List.iter [ File; Stdout ] ~f:(fun destination ->
        check_case ~destination ~location_format:"beg_pos" ~want_location:true;
        check_case ~destination ~location_format:"no_location" ~want_location:false)
