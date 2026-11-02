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
