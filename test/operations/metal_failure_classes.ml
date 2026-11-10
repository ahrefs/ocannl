(* gh-ocannl-1436: a Metal compile whose compiler service crashed is a typed decline of that kernel.

   On mac-studio (M4 Max, macOS 27.2) the unparallelized whole-routine gpt2_mini_train kernel that a
   pre-gh-ocannl-1340 search compiled as its arm A baseline aborted [MTLCompilerService] in AGX
   instruction emission on every attempt -- Metal's first and its own retries alike -- and
   [newComputePipelineStateWithFunction:] failed with [XPC_ERROR_CONNECTION_INTERRUPTED]. Untyped,
   strict classification made that [Fatal], so one kernel Apple's compiler crashed on ended the
   whole arm. gh-ocannl-1340 stopped compiling that baseline, so the crash no longer has a trigger
   on demand: the classes are pinned here over the binding's texts, the crash text verbatim as
   captured, with the threadgroup overflow (gh-ocannl-1226) that shares the classifier and the XPC
   failure that is not a crash as controls. GPU-free: [Metal_failure] lives outside the
   [metal]-gated backend.

   The cause the backend raises is pinned too: a containable [Compiler_bug] decline at its own
   stage, classified -- not [Fatal] -- under strict classification and [User_schedule] provenance at
   both places Metal compiles (the library, [Backend_compile]; the pipeline, [Backend_link]), while
   the same text left untyped stays [Fatal] there. *)

open Base
open Verdict.Claims
module MF = Ir.Metal_failure
module SO = Ir.Schedule_outcome

let crash =
  "newComputePipelineStateWithFunction:options:reflection:error: failed: Compilation failed due to \
   an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple \
   retries."

let cases =
  [
    ("the captured pipeline crash", crash, MF.Compiler_service_crash);
    ( "a library-compile crash (constructed: the binding's prefix, the observed description)",
      "newLibraryWithSource:options:error: failed: Compilation failed due to an interrupted \
       connection: XPC_ERROR_CONNECTION_INTERRUPTED.",
      MF.Compiler_service_crash );
    ( "a threadgroup overflow",
      "newComputePipelineStateWithFunction:options:reflection:error: failed: Threadgroup memory \
       size (32772) exceeds the maximum threadgroup memory allowed (32768)",
      MF.Threadgroup_overflow { requested = 32772; limit = 32768 } );
    ( "an unreachable compiler service",
      "newComputePipelineStateWithFunction:options:reflection:error: failed: Compilation failed \
       due to an invalid connection: XPC_ERROR_CONNECTION_INVALID.",
      MF.Unrecognized );
    ( "an unrelated pipeline failure",
      "newComputePipelineStateWithFunction:options:reflection:error: failed: Function \
       ocannl_kernel is not a compute function.",
      MF.Unrecognized );
  ]

let () =
  List.iter cases ~f:(fun (label, msg, expected) ->
      let got = MF.classify_pipeline_failure msg in
      Stdio.eprintf "%s: %s (not part of the golden)\n" label
        (Sexp.to_string (MF.sexp_of_pipeline_failure got));
      p
        (Printf.sprintf "%s is %s" label (Sexp.to_string (MF.sexp_of_pipeline_failure expected)))
        (MF.equal_pipeline_failure got expected));
  List.iter cases ~f:(fun (label, msg, expected) ->
      p
        (Printf.sprintf "%s: the library-compile test agrees with the pipeline class" label)
        (Bool.equal (MF.compiler_service_crash msg)
           (MF.equal_pipeline_failure expected MF.Compiler_service_crash)))

let describe = function
  | Ok () -> "succeeded"
  | Error (SO.Classified { phase; cause; execution_effect }) ->
      Printf.sprintf "classified at %s, %s, %s"
        (Sexp.to_string (SO.sexp_of_phase phase))
        (Sexp.to_string (SO.sexp_of_rejection_key (SO.key_of_cause cause)))
        (Sexp.to_string (SO.sexp_of_execution_effect execution_effect))
  | Error (SO.Fatal { exn; _ }) -> "FATAL " ^ Exn.to_string exn

let protect phase f =
  SO.protect ~strict:true ~classify_backend:(fun _ _ -> None) ~provenance:SO.User_schedule ~phase f

let () =
  let kernel = "ocannl_gh1436_kernel" in
  let cause =
    MF.compiler_service_crash_cause ~backend:"metal" ~request:"the pipeline" ~kernel crash
  in
  Stdio.printf "the crash's key: %s\n"
    (Sexp.to_string (SO.sexp_of_rejection_key (SO.key_of_cause cause)));
  p "the crash is a Compiler_bug rejection at its own stage, not the compiler's diagnosed one"
    (match cause with
    | SO.Backend_rejected { backend = "metal"; stage; severity = SO.Compiler_bug; _ } ->
        String.equal stage MF.compiler_crash_stage && not (String.equal stage "compiler")
    | _ -> false);
  let detail = SO.detail_of_cause cause in
  p "the detail names the kernel" (String.is_substring detail ~substring:kernel);
  p "the detail carries the driver's report" (String.is_substring detail ~substring:crash);
  p "the detail says Metal retried, as the report does"
    (String.is_substring detail ~substring:"Metal's own retries");
  p "the detail does not claim an OCANNL codegen bug"
    (not (String.is_substring detail ~substring:Ir.C_syntax.codegen_bug_claim_prefix));
  List.iter [ SO.Backend_compile; SO.Backend_link ] ~f:(fun phase ->
      let phase_name = Sexp.to_string (SO.sexp_of_phase phase) in
      p
        (Printf.sprintf "the crash is containable at %s" phase_name)
        (not (SO.uncontainable phase cause));
      let typed = protect phase (fun () -> raise (SO.Cause_at (phase, cause))) in
      Stdio.printf "typed at %s: %s\n" phase_name (describe typed);
      p
        (Printf.sprintf "typed at %s, the crash is a decline that wrote nothing" phase_name)
        (match typed with
        | Error (SO.Classified { phase = got; execution_effect = SO.No_device_writes; _ }) ->
            SO.equal_phase got phase
        | Ok () | Error _ -> false);
      let untyped = protect phase (fun () -> failwith crash) in
      Stdio.eprintf "untyped at %s: %s (not part of the golden)\n" phase_name (describe untyped);
      p
        (Printf.sprintf "untyped at %s, the same text is Fatal (the control)" phase_name)
        (match untyped with Error (SO.Fatal _) -> true | Ok () | Error (SO.Classified _) -> false))

(* gh-ocannl-2089: a failed command buffer's first report of a GPU hang is the candidate's typed
   decline, one that may have written buffers; everything else a failed buffer reports stays
   unclassified, as every Metal runtime failure was before. The hang text is the one
   metal_gpu_hang_failure captured (five runs, M4 Max, macOS 27.2); the ignored and discarded
   descriptions are constructed from the states its notes name, with the driver's codes for them,
   which no run here has printed. Every case goes through strict classification with [Candidate]
   provenance, the tuner's timing boundary. *)
let hang_error = "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)"

let hang =
  {
    MF.what = "kernel ocannl_gh2089_kernel";
    runs_kernels = true;
    error = Some hang_error;
    first_report = true;
  }

let classify_metal = MF.classify_command_buffer_failure ~backend:"metal"

let at phase exn =
  SO.protect ~strict:true ~classify_backend:classify_metal ~provenance:SO.Candidate ~phase
    (fun () -> raise exn)

let () =
  p "the captured hang text reports a GPU hang" (MF.reports_gpu_hang hang_error);
  p "the report renders as the backend's text did before it was typed"
    (String.equal
       (Stdlib.Printexc.to_string (MF.Command_buffer_failed hang))
       ("Metal: the command buffer of kernel ocannl_gh2089_kernel failed: " ^ hang_error
      ^ ". The device takes no more work in this process."));
  p "a report without error text says so"
    (String.is_substring
       (MF.command_buffer_report { hang with error = None })
       ~substring:"Metal gave no error text");
  List.iter [ SO.Sync; SO.Launch ] ~f:(fun phase ->
      let phase_name = Sexp.to_string (SO.sexp_of_phase phase) in
      let got = at phase (MF.Command_buffer_failed hang) in
      Stdio.printf "the hang's first report at %s: %s\n" phase_name (describe got);
      p
        (Printf.sprintf
           "at %s, the hang's first report is an Expected gpu_hang decline that may have written"
           phase_name)
        (match got with
        | Error
            (SO.Classified
               {
                 phase = got_phase;
                 cause =
                   SO.Backend_rejected { backend = "metal"; stage; severity = SO.Expected; detail };
                 execution_effect = SO.Writes_may_have_occurred;
               }) ->
            SO.equal_phase got_phase phase
            && String.equal stage MF.gpu_hang_stage
            && String.is_substring detail ~substring:hang_error
            && String.is_substring detail ~substring:"kernel ocannl_gh2089_kernel"
        | Ok () | Error _ -> false));
  let controls =
    [
      ( "a later report of the same hang",
        MF.Command_buffer_failed { hang with first_report = false } );
      ( "a hang reported for a buffer that ran no kernel",
        MF.Command_buffer_failed { hang with what = "a zero fill"; runs_kernels = false } );
      ( "an ignored submission (constructed)",
        MF.Command_buffer_failed
          {
            hang with
            error =
              Some
                "Ignored (for causing prior/excessive GPU errors) \
                 (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)";
          } );
      ( "a discarded victim (constructed)",
        MF.Command_buffer_failed
          {
            hang with
            error =
              Some
                "Discarded (victim of GPU error/recovery) \
                 (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)";
          } );
      ("a failure without error text", MF.Command_buffer_failed { hang with error = None });
      ("the hang's report text in a plain Failure", Failure (MF.command_buffer_report hang));
    ]
  in
  List.iter controls ~f:(fun (label, exn) ->
      List.iter [ SO.Sync; SO.Launch ] ~f:(fun phase ->
          let phase_name = Sexp.to_string (SO.sexp_of_phase phase) in
          p
            (Printf.sprintf "%s at %s stays unclassified, so Fatal" label phase_name)
            (match at phase exn with
            | Error (SO.Fatal _) -> true
            | Ok () | Error (SO.Classified _) -> false)));
  p_all "the hang's first report outside Launch and Sync is not the backend's to classify"
    [ SO.Transform; SO.Backend_compile; SO.Backend_link; SO.Preflight ] ~f:(fun phase ->
      Option.is_none (classify_metal phase (MF.Command_buffer_failed hang)))
