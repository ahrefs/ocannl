(* gh-ocannl-1142: exercise both post-compile failures through the real Context and autotune
   boundaries with permissive classification. The compiler fixture either reports success without an
   artifact, or writes an artifact; a private PATH entry makes codesign exit 42. No compiler or
   signing service is needed. Windows pins the classification but skips these Unix fixtures. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let stage = Stdlib.Sys.argv.(1)
let phase_name phase = Sexp.to_string (Outcome.sexp_of_phase phase)

let describe_failure = function
  | Outcome.Classified { phase; cause; _ } ->
      Printf.sprintf "Classified at %s: %s" (phase_name phase)
        (Sexp.to_string (Outcome.sexp_of_rejection_key (Outcome.key_of_cause cause)))
  | Outcome.Fatal { exn; phase; _ } ->
      Printf.sprintf "Fatal at %s: %s" (phase_name phase) (Stdlib.Printexc.exn_slot_name exn)

let run () =
  let ctx = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name ctx);
  let compiler_command = Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" in
  p "strict failure classification is off"
    (not (Utils.get_global_flag ~default:true ~arg_name:"strict_failure_classification"));
  let x = TDSL.range 16 in
  let%op y = exp (x /. 16.) in
  let comp = Train.forward y in
  (* 1. The containment-aware compile, as a search candidate would run it. *)
  let fatal =
    if Sys.win32 then None
    else
      match
        Context.compile_outcome ~provenance:Outcome.Candidate ~candidate:"post-compile probe" ctx
          comp Ir.Indexing.Empty
      with
      | Ok _ ->
          Stdio.eprintf "candidate compile: succeeded\n%!";
          None
      | Error failure -> (
          Stdio.eprintf "candidate compile: %s\n%!" (describe_failure failure);
          match failure with Outcome.Fatal fatal -> Some fatal | Outcome.Classified _ -> None)
  in
  let rejection =
    match fatal with
    | Some { cause = Some (Outcome.Backend_rejected { backend; stage; severity; detail }); _ } ->
        Stdio.eprintf "post-compile detail:\n%s\n%!" detail;
        Some (backend, stage, severity, detail)
    | _ -> None
  in
  let rejection_has f = Option.exists rejection ~f in
  (* 2. The public compile renders the cause through the exception contract. *)
  let compile_exn =
    if Sys.win32 then None
    else
      match Context.compile ctx comp Ir.Indexing.Empty with
      | _ -> None
      | exception exn ->
          Stdio.eprintf "Context.compile raised: %s\n%!" (Exn.to_string exn);
          Some exn
  in
  (* 3. Autotune does not absorb it: the baseline's post-compile failure ends the call, with a
     report. *)
  let report = ref None in
  let tune_raised =
    if Sys.win32 then false
    else
      match
        Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
          ~report:(fun r -> report := Some r)
          ctx comp Ir.Indexing.Empty
      with
      | _ -> false
      | exception exn ->
          Stdio.eprintf "Autotune.tune raised: %s\n%!" (Stdlib.Printexc.exn_slot_name exn);
          true
  in
  let pre_search_phase =
    match !report with
    | Some { Autotune.outcome = Autotune.Pre_search_failure { phase; _ }; _ } -> Some phase
    | Some _ | None -> None
  in
  let claim =
    gated ~aggregation:`Environment ~when_:(not Sys.win32)
      ~on:"Windows, where the Unix shell fixtures cannot run"
  in
  claim "the candidate compile is fatal, not a contained decline" (Option.is_some fatal);
  claim "the fatal failure is at Backend_link"
    (Option.exists fatal ~f:(fun f -> Outcome.equal_phase f.phase Outcome.Backend_link));
  claim "the fatal failure carries its Backend_rejected cause" (Option.is_some rejection);
  claim "the rejecting backend is cc" (rejection_has (fun (b, _, _, _) -> String.equal b "cc"));
  claim "the stage matches the injected post-compile failure"
    (rejection_has (fun (_, stage, _, _) -> String.equal stage Stdlib.Sys.argv.(1)));
  claim "the severity is Compiler_bug"
    (rejection_has (fun (_, _, severity, _) -> Outcome.equal_severity severity Outcome.Compiler_bug));
  claim "the detail names the missing artifact or codesign exit status"
    (rejection_has (fun (_, _, _, detail) ->
         String.is_substring detail
           ~substring:
             (if String.equal stage Outcome.artifact_missing_stage then
                "not found after successful compilation"
              else "exit code 42")));
  claim "the detail carries the compile/link command"
    (rejection_has (fun (_, _, _, detail) -> String.is_substring detail ~substring:compiler_command));
  claim "Context.compile raises Invalid_argument with the post-compile diagnostic"
    (match compile_exn with
    | Some (Invalid_argument msg) ->
        String.is_substring msg ~substring:"Cc_backend.c_compile_and_load"
    | _ -> false);
  claim "Autotune.tune raises" tune_raised;
  claim "autotune reports a pre-search failure at Backend_link"
    (Option.exists pre_search_phase ~f:(Outcome.equal_phase Outcome.Backend_link))

let () =
  List.iter [ Outcome.artifact_missing_stage; Outcome.codesign_stage ] ~f:(fun stage ->
      let cause =
        Outcome.Backend_rejected
          {
            backend = "cc";
            stage;
            severity = Outcome.Compiler_bug;
            detail = "classification probe";
          }
      in
      pf "%s at Backend_link is uncontainable" stage
        (Outcome.uncontainable Outcome.Backend_link cause);
      pf "%s at Backend_compile remains containable" stage
        (not (Outcome.uncontainable Outcome.Backend_compile cause)));
  if Sys.win32 then run ()
  else
    let dir = Stdlib.Filename.temp_file "ocannl gh1142 " "" in
    Stdlib.Sys.remove dir;
    Unix.mkdir dir 0o700;
    let codesign = Stdlib.Filename.concat dir "codesign" in
    Stdio.Out_channel.write_all codesign ~data:"#!/bin/sh\nexit 42\n";
    Unix.chmod codesign 0o700;
    let path = Option.value (Sys.getenv "PATH") ~default:"" in
    Unix.putenv "PATH" (dir ^ ":" ^ path);
    Exn.protect ~f:run ~finally:(fun () ->
        Unix.putenv "PATH" path;
        Stdlib.Sys.remove codesign;
        Unix.rmdir dir)
