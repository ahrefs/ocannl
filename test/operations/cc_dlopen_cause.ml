(* gh-ocannl-1077: a cc kernel that compiles but does not load -- [dlopen] fails on an undefined
   symbol -- is classified at [Backend_link] as [Backend_rejected { stage = "dlopen"; severity =
   Compiler_bug }], carrying [dlerror()] and the compile/link command, instead of escaping as a raw
   [Dl.DL_error]. And it is a hard error, never a candidate decline: a symbol the kernel references
   and nothing supplies is an OCANNL link bug (gh-ocannl-1045's missing [-lm] was one), and a search
   that absorbed it would quietly prefer the candidates that happen not to reach the symbol.

   The dune rule manufactures the failure through the compiler command:
   [-Dexpf=ocannl_gh1077_undefined_expf] renames the [expf] the kernel calls (and its [<math.h>]
   declaration with it), so the kernel compiles, links as a shared object with the reference
   unresolved, and dies at [dlopen]'s [RTLD_NOW] binding. glibc's [<bits/mathcalls.h>] pastes the
   declared name into a [__DECL_SIMD_<name>] macro, which the rule therefore defines empty for the
   new name; elsewhere that define is inert. The rule also runs with
   [strict_failure_classification=false], the setting under which the compile-side containment
   absorbs an unclassified exception as a decline: before gh-ocannl-1077 the raw [Dl.DL_error] was
   absorbed there (strict classification made it fatal, but untyped and at the wrong phase), so the
   containment claims discriminate.

   gh-ocannl-1142 adds two more modes to this same boundary probe. [artifact_missing] uses a
   compiler stub that reports success without producing a library; [codesign] uses a stub that
   writes an artifact and a private PATH shim that exits 42. Only the codesign mode changes PATH.
   The unit test [test_schedule_outcome] pins all three stages over every provenance and strictness.

   Windows cannot manufacture the dlopen failure: a PE DLL link resolves every import, so the
   undefined reference fails the compile before there is anything to load, and the claims skip
   there.

   gh-ocannl-1717 splits the severity: an undefined symbol stays a [Compiler_bug], while a missing
   artifact is the environment's refusal ([Environment], rendered as [Failure]), and so is the
   fourth mode, [code_signing]: the stub runs the real compiler, then strips the library's code
   signature, so arm64 macOS's loader refuses it at [dlopen] as it refused the transient
   [schedule_conv_gemm] library (a host that enforces no signatures loads it, and the claims skip).
   [cc_load_refusal_classes] pins the classes over canned loader texts. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let undefined_symbol = "ocannl_gh1077_undefined_expf"

(* The mode names the injected failure; the stage is the one it is rejected at. *)
let mode = Stdlib.Sys.argv.(1)

(* arm64 macOS refuses to map unsigned code; other hosts load an unsigned library. Read off the
   host, never off the run under test. *)
let enforces_code_signatures =
  let ask command =
    try
      let ic = Unix.open_process_in command in
      let answer = String.strip (Stdio.In_channel.input_all ic) in
      ignore (Unix.close_process_in ic : Unix.process_status);
      answer
    with _ -> ""
  in
  (not Sys.win32) && String.equal (ask "uname -s") "Darwin" && String.equal (ask "uname -m") "arm64"

let expected_stage, detail_substring, expected_severity, applicable, skip_reason =
  let unix_fixture = "Windows, where the Unix shell fixtures cannot run" in
  match mode with
  | "dlopen" ->
      ( Outcome.dlopen_stage,
        undefined_symbol,
        Outcome.Compiler_bug,
        not Sys.win32,
        "a PE DLL link, which refuses the undefined reference before dlopen" )
  | "artifact_missing" ->
      ( Outcome.artifact_missing_stage,
        "not found after successful compilation",
        Outcome.Environment,
        not Sys.win32,
        unix_fixture )
  | "codesign" ->
      (Outcome.codesign_stage, "exit code 42", Outcome.Compiler_bug, not Sys.win32, unix_fixture)
  | "code_signing" ->
      ( Outcome.dlopen_stage,
        "code signature",
        Outcome.Environment,
        enforces_code_signatures,
        "a host whose loader enforces no code signatures (only arm64 macOS does)" )
  | mode -> invalid_arg ("unknown post-compile probe mode: " ^ mode)

let severity_name severity = Sexp.to_string (Outcome.sexp_of_severity severity)

let exception_name severity =
  match
    Outcome.exception_of_cause
      (Outcome.Backend_rejected { backend = "cc"; stage = ""; severity; detail = "" })
  with
  | Invalid_argument _ -> "Invalid_argument"
  | Failure _ -> "Failure"
  | Utils.User_error _ -> "User_error"
  | exn -> Stdlib.Printexc.exn_slot_name exn

let raised_expected severity exn =
  match (exception_name severity, exn) with
  | "Invalid_argument", Invalid_argument msg | "Failure", Failure msg ->
      String.is_substring msg ~substring:detail_substring
  | _ -> false

let phase_name phase = Sexp.to_string (Outcome.sexp_of_phase phase)

let describe_failure = function
  | Outcome.Classified { phase; cause; _ } ->
      Printf.sprintf "Classified at %s: %s" (phase_name phase)
        (Sexp.to_string (Outcome.sexp_of_rejection_key (Outcome.key_of_cause cause)))
  | Outcome.Fatal { exn; phase; _ } ->
      Printf.sprintf "Fatal at %s: %s" (phase_name phase) (Stdlib.Printexc.exn_slot_name exn)

let run () =
  let artifacts = ref [] in
  let directory = Stdlib.Filename.get_temp_dir_name () in
  let snapshot () =
    Stdlib.Sys.readdir directory |> Array.to_list |> List.sort ~compare:String.compare
  in
  let before = snapshot () in
  let debug = Utils.settings.output_debug_files_in_build_directory in
  let dll_output =
    Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory"
  in
  Stdio.printf "library artifacts requested: %s\n" (if dll_output then "yes" else "no");
  Stdio.printf "debug artifacts requested: %s\n" (if debug then "yes" else "no");
  let ctx = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name ctx);
  let compiler_command = Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" in
  if String.equal mode "dlopen" then
    p "the compiler command renames expf to the undefined symbol"
      (String.is_substring compiler_command ~substring:("-Dexpf=" ^ undefined_symbol));
  p "strict failure classification is off"
    (not (Utils.registered_flag ~arg_name:"strict_failure_classification"));
  let x = TDSL.range 16 in
  let%op y = exp (x /. 16.) in
  let comp = Train.forward y in
  (* 1. The containment-aware compile, as a search candidate would run it. *)
  let fatal =
    if not applicable then None
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
  artifacts := snapshot ();
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
    if not applicable then None
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
    if not applicable then false
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
  let claim = gated ~aggregation:`Environment ~when_:applicable ~on:skip_reason in
  let added =
    List.filter !artifacts ~f:(fun path -> not (List.mem before path ~equal:String.equal))
  in
  (* Library output and debug files keep what they ask for in [build_files/], not here
     (gh-ocannl-1247, gh-ocannl-1335). *)
  Stdio.eprintf "added to the temp directory (not part of the golden): %s\n%!"
    (String.concat ~sep:" " added);
  gated_all ~aggregation:`Environment ~when_:applicable ~on:skip_reason
    "every file in the temp directory after the rejected compile was there before it" !artifacts
    ~f:(fun path -> List.mem before path ~equal:String.equal);
  claim "the unrelated temporary file is preserved"
    (String.equal
       (Stdio.In_channel.read_all (Stdlib.Filename.concat directory "unrelated.so"))
       "unrelated");
  claim "the candidate compile is fatal, not a contained decline" (Option.is_some fatal);
  claim "the fatal failure is at Backend_link"
    (Option.exists fatal ~f:(fun f -> Outcome.equal_phase f.phase Outcome.Backend_link));
  claim "the fatal failure carries its Backend_rejected cause" (Option.is_some rejection);
  claim "the rejecting backend is cc" (rejection_has (fun (b, _, _, _) -> String.equal b "cc"));
  claim "the stage matches the injected post-compile failure"
    (rejection_has (fun (_, stage, _, _) -> String.equal stage expected_stage));
  claim
    ("the severity is " ^ severity_name expected_severity)
    (rejection_has (fun (_, _, severity, _) -> Outcome.equal_severity severity expected_severity));
  claim "the detail carries the injected failure diagnostic"
    (rejection_has (fun (_, _, _, detail) -> String.is_substring detail ~substring:detail_substring));
  claim "the detail carries the compile/link command"
    (rejection_has (fun (_, _, _, detail) -> String.is_substring detail ~substring:compiler_command));
  claim
    ("Context.compile raises " ^ exception_name expected_severity
   ^ " with the post-compile diagnostic")
    (Option.exists compile_exn ~f:(raised_expected expected_severity));
  claim "Autotune.tune raises" tune_raised;
  claim "autotune reports a pre-search failure at Backend_link"
    (Option.exists pre_search_phase ~f:(Outcome.equal_phase Outcome.Backend_link))

let run_in_private_directory () =
  let original = Stdlib.Filename.get_temp_dir_name () in
  let dir = Stdlib.Filename.temp_file "ocannl-gh1189-" "" in
  Stdlib.Sys.remove dir;
  Unix.mkdir dir 0o700;
  Stdlib.Filename.set_temp_dir_name dir;
  Stdio.Out_channel.write_all (Stdlib.Filename.concat dir "unrelated.so") ~data:"unrelated";
  Exn.protect ~f:run ~finally:(fun () ->
      Stdlib.Filename.set_temp_dir_name original;
      Array.iter (Stdlib.Sys.readdir dir) ~f:(fun file ->
          Stdlib.Sys.remove (Stdlib.Filename.concat dir file));
      Unix.rmdir dir)

let () =
  if Sys.win32 || not (String.equal mode "codesign") then run_in_private_directory ()
  else
    let dir = Stdlib.Filename.temp_file "ocannl gh1142 " "" in
    Stdlib.Sys.remove dir;
    Unix.mkdir dir 0o700;
    let codesign = Stdlib.Filename.concat dir "codesign" in
    Stdio.Out_channel.write_all codesign ~data:"#!/bin/sh\nexit 42\n";
    Unix.chmod codesign 0o700;
    let path = Option.value (Sys.getenv "PATH") ~default:"" in
    Unix.putenv "PATH" (dir ^ ":" ^ path);
    Exn.protect ~f:run_in_private_directory ~finally:(fun () ->
        Unix.putenv "PATH" path;
        Stdlib.Sys.remove codesign;
        Unix.rmdir dir)
