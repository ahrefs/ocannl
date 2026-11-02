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
