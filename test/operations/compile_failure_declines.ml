(* ahrefs/ocannl#1387: generated code that the backend's compiler rejects is a typed decline of that
   kernel on every backend — [Backend_rejected] at stage ["compiler"], a [Compiler_bug] that wrote
   nothing — and not an untyped exception. Under [User_schedule] provenance (the isolated per-kernel
   instrument's: [Bench_harness.compile_shipped_kernel]) an untyped compile failure is classified
   [Fatal], so before the fix one kernel hiprtc or Metal refused ended the whole per-kernel table
   partway, while cc (a typed [Cause_at] at the compiler call) and CUDA (its [classify_failure]
   recognizing [NVRTC_ERROR_COMPILATION]) declined just that kernel.

   The rejected code is a staged statement naming an identifier nothing declares, which no C-family
   compiler accepts; the rest of the kernel is ordinary. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  let tn = (Ll_test.node_factory ~first_id:13870 ~dims:[| 1 |] ()) "cfd_out" in
  Ll_test.materialize tn;
  let cell = [| Ll_test.fixed 0 |] in
  let o =
    Ll_test.optimize_scoped ~materialized:[ tn ] ~name:"cfd_rejected"
      ~raw:(Ll_test.set tn cell (Ir.Low_level.Constant 1.0))
      (Ir.Low_level.Staged_compilation
         (fun () -> PPrint.string "cfd_out[0] = ocannl_gh1387_undeclared_identifier;"))
  in
  let outcome =
    Context.compile_outcome ~name:"cfd_rejected" ~prelowered:o
      ~lowered_transform:(fun x -> [ x ])
      ~provenance:Outcome.User_schedule (Lazy.force Ll_test.base_ctx) Ir.Assignments.empty_comp
      Ir.Indexing.Empty
  in
  (match outcome with
  | Ok _ -> Stdio.eprintf "outcome: compiled (not part of the golden)\n%!"
  | Error (Outcome.Classified { phase; cause; execution_effect }) ->
      Stdio.eprintf "outcome: classified at %s, %s, %s (not part of the golden)\n%!"
        (Sexp.to_string_hum (Outcome.sexp_of_phase phase))
        (Sexp.to_string_hum (Outcome.sexp_of_rejection_key (Outcome.key_of_cause cause)))
        (Sexp.to_string_hum (Outcome.sexp_of_execution_effect execution_effect))
  | Error (Outcome.Fatal { exn; _ }) ->
      Stdio.eprintf "outcome: FATAL %s (not part of the golden)\n%!"
        (String.prefix (Exn.to_string exn) 400));
  p
    "a compiler-rejected kernel is a typed decline: the backend's own compiler stage, a codegen \
     bug, no device writes"
    (match outcome with
    | Error
        (Outcome.Classified
           {
             cause =
               Outcome.Backend_rejected { stage = "compiler"; severity = Outcome.Compiler_bug; _ };
             execution_effect = Outcome.No_device_writes;
             _;
           }) ->
        true
    | Ok _ | Error _ -> false)
