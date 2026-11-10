open Base

type pipeline_failure =
  | Threadgroup_overflow of { requested : int; limit : int }
  | Compiler_service_crash
  | Unrecognized
[@@deriving sexp_of, equal]

let compiler_service_crash msg =
  String.is_substring msg ~substring:"XPC_ERROR_CONNECTION_INTERRUPTED"

(* The driver's sentence, as [newComputePipelineStateWithFunction:] reports an over-limit
   threadgroup allocation (gh-ocannl-1226). *)
let threadgroup_overflow msg =
  Option.bind (String.substr_index msg ~pattern:"Threadgroup memory size (") ~f:(fun pos ->
      try
        Some
          (Stdlib.Scanf.sscanf (String.drop_prefix msg pos)
             "Threadgroup memory size (%d) exceeds the maximum threadgroup memory allowed (%d)"
             (fun requested limit -> (requested, limit)))
      with Stdlib.Scanf.Scan_failure _ | End_of_file | Failure _ -> None)

let classify_pipeline_failure msg =
  match threadgroup_overflow msg with
  | Some (requested, limit) -> Threadgroup_overflow { requested; limit }
  | None -> if compiler_service_crash msg then Compiler_service_crash else Unrecognized

let compiler_crash_stage = "compiler_crash"

let compiler_service_crash_cause ~backend ~request ~kernel report =
  let retried =
    if String.is_substring report ~substring:"after multiple retries" then
      ", on Metal's own retries too"
    else ""
  in
  Schedule_outcome.Backend_rejected
    {
      backend;
      stage = compiler_crash_stage;
      severity = Schedule_outcome.Compiler_bug;
      detail =
        Printf.sprintf
          "Metal: the Metal compiler service crashed compiling %s for %s%s. This is a defect of \
           Apple's compiler on this kernel, not a rejection of the generated code: seen on every \
           attempt for one very large kernel, an unparallelized whole routine (gh-ocannl-1436). \
           The service's crash report is \
           ~/Library/Logs/DiagnosticReports/MTLCompilerService-*.ips; rerun with \
           --ocannl_output_debug_files_in_build_directory=true to write the kernel's .metal source.\n\
           Driver: %s"
          request kernel retried report;
    }

type command_buffer_failure = {
  what : string;
  runs_kernels : bool;
  error : string option;
  first_report : bool;
}
[@@deriving sexp_of, equal]

exception Command_buffer_failed of command_buffer_failure

let command_buffer_report { what; error; _ } =
  let error = Option.value error ~default:"Metal gave no error text" in
  Printf.sprintf
    "Metal: the command buffer of %s failed: %s. The device takes no more work in this process."
    what error

let () =
  Stdlib.Printexc.register_printer (function
    | Command_buffer_failed failure -> Some (command_buffer_report failure)
    | _ -> None)

(* The driver's code for a kernel the GPU watchdog killed, as it ends Metal's error description:
   [Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)], captured verbatim on an
   M4 Max (macOS 27.2, test/operations/metal_gpu_hang_failure, gh-ocannl-2083). The token rather
   than the English sentence, which [localizedDescription] translates. Observed tokens only
   (gh-ocannl-2089): a timeout or an older [kIOAccel] spelling joins once a run has shown it. *)
let gpu_hang_token = "kIOGPUCommandBufferCallbackErrorHang)"
let reports_gpu_hang error = String.is_substring error ~substring:gpu_hang_token
let gpu_hang_stage = "gpu_hang"

let classify_command_buffer_failure ~backend (phase : Schedule_outcome.phase) exn =
  match (phase, exn) with
  | ( (Launch | Sync),
      Command_buffer_failed
        ({ first_report = true; runs_kernels = true; error = Some error; _ } as failure) )
    when reports_gpu_hang error ->
      Some
        {
          Schedule_outcome.phase;
          cause =
            Schedule_outcome.Backend_rejected
              {
                backend;
                stage = gpu_hang_stage;
                severity = Schedule_outcome.Expected;
                detail =
                  command_buffer_report failure
                  ^ " The GPU watchdog killed the kernel: the candidate's schedule ran too long, \
                     not a codegen bug.";
              };
          execution_effect = Schedule_outcome.Writes_may_have_occurred;
        }
  | _ -> None
