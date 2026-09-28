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

   Windows cannot manufacture the failure: a PE DLL link resolves every import, so the undefined
   reference fails the compile before there is anything to load, and the claims skip there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let undefined_symbol = "ocannl_gh1077_undefined_expf"
let phase_name phase = Sexp.to_string (Outcome.sexp_of_phase phase)

let describe_failure = function
  | Outcome.Classified { phase; cause; _ } ->
      Printf.sprintf "Classified at %s: %s" (phase_name phase)
        (Sexp.to_string (Outcome.sexp_of_rejection_key (Outcome.key_of_cause cause)))
  | Outcome.Fatal { exn; phase; _ } ->
      Printf.sprintf "Fatal at %s: %s" (phase_name phase) (Stdlib.Printexc.exn_slot_name exn)

let () =
  let ctx = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name ctx);
  let compiler_command = Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" in
  p "the compiler command renames expf to the undefined symbol"
    (String.is_substring compiler_command ~substring:("-Dexpf=" ^ undefined_symbol));
  p "strict failure classification is off"
    (not (Utils.get_global_flag ~default:true ~arg_name:"strict_failure_classification"));
  let x = TDSL.range 16 in
  let%op y = exp (x /. 16.) in
  let comp = Train.forward y in
  (* 1. The containment-aware compile, as a search candidate would run it. *)
  let fatal =
    match
      Context.compile_outcome ~provenance:Outcome.Candidate ~candidate:"dlopen probe" ctx comp
        Ir.Indexing.Empty
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
        Stdio.eprintf "dlopen detail:\n%s\n%!" detail;
        Some (backend, stage, severity, detail)
    | _ -> None
  in
  let rejection_has f = Option.exists rejection ~f in
  (* 2. The public compile renders the cause through the exception contract. *)
  let compile_exn =
    match Context.compile ctx comp Ir.Indexing.Empty with
    | _ -> None
    | exception exn ->
        Stdio.eprintf "Context.compile raised: %s\n%!" (Exn.to_string exn);
        Some exn
  in
  (* 3. Autotune does not absorb it: the baseline's dlopen failure ends the call, with a report. *)
  let report = ref None in
  let tune_raised =
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
      ~on:"a PE DLL link, which refuses the undefined reference before dlopen"
  in
  claim "the candidate compile is fatal, not a contained decline" (Option.is_some fatal);
  claim "the fatal failure is at Backend_link"
    (Option.exists fatal ~f:(fun f -> Outcome.equal_phase f.phase Outcome.Backend_link));
  claim "the fatal failure carries its Backend_rejected cause" (Option.is_some rejection);
  claim "the rejecting backend is cc" (rejection_has (fun (b, _, _, _) -> String.equal b "cc"));
  claim "the stage is dlopen"
    (rejection_has (fun (_, stage, _, _) -> String.equal stage Outcome.dlopen_stage));
  claim "the severity is Compiler_bug"
    (rejection_has (fun (_, _, severity, _) -> Outcome.equal_severity severity Outcome.Compiler_bug));
  claim "the detail carries dlerror's unresolved symbol"
    (rejection_has (fun (_, _, _, detail) -> String.is_substring detail ~substring:undefined_symbol));
  claim "the detail carries the compile/link command"
    (rejection_has (fun (_, _, _, detail) -> String.is_substring detail ~substring:compiler_command));
  claim "Context.compile raises Invalid_argument naming dlopen"
    (match compile_exn with
    | Some (Invalid_argument msg) -> String.is_substring msg ~substring:"dlopen"
    | _ -> false);
  claim "Autotune.tune raises" tune_raised;
  claim "autotune reports a pre-search failure at Backend_link"
    (Option.exists pre_search_phase ~f:(Outcome.equal_phase Outcome.Backend_link))
