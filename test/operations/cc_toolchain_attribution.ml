(* gh-ocannl-1350: a cc compile failure is blamed on OCANNL's generated code only when the toolchain
   configuration compiles a trivial kernel. When it does not -- an explicit flag the compiler
   rejects, a compiler command that names no compiler -- the failure is a typed configuration error
   naming the setting, and it ends a search rather than declining one candidate: the toolchain would
   reject every candidate alike.

   Each dune rule breaks the configuration one way. [arch_flags] passes a target flag no compiler
   knows, [include_flag] a flag forcing in a header that does not exist (a missing header is
   otherwise the generic installation error of ahrefs/ocannl#1387, which must not pre-empt the
   setting's name), [compiler_command] names a compiler that does not exist, and [codegen] -- the
   control -- keeps the toolchain intact but renames [float] to an undeclared type in the compiler
   command, so only the generated code fails to compile, and a trivial kernel (which declares no
   float) still does: the bug claim stays. [macro_label] is a second such control: the compiler
   command defines the computation's node label as [__LINE__], so the node's parameter declaration
   fails to compile -- what a node labelled [__LINE__] did before gh-ocannl-1338 renamed such
   labels. clang then quotes the routine's [void <name>(] line under its [note: to match this '('],
   which must not make the failure the routine NAME's: only an error's quoted lines blame the
   name. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let mode = Stdlib.Sys.argv.(1)

(* The setting each configuration mode breaks. *)
let culprit =
  match mode with
  | "arch_flags" | "include_flag" -> Some "cc_backend_arch_flags"
  | "compiler_command" -> Some "cc_backend_compiler_command"
  | "codegen" | "macro_label" -> None
  | other -> invalid_arg ("cc_toolchain_attribution: unknown mode " ^ other)

(* The settings a toolchain failure can name: the compiler command and every setting a kernel
   compile passes a flag for. *)
let settings =
  [
    "cc_backend_compiler_command";
    "cc_backend_optimization_level";
    "cc_backend_arch_flags";
    "cc_backend_simd_flags";
    "cc_backend_fast_math";
    "cc_backend_fp_contract";
    "cc_parallel_grid";
  ]

type rejection = { backend : string; stage : string; severity : Outcome.severity; detail : string }

(* The text between two section headings of a failure detail. *)
let section detail ~from ~until =
  Option.bind (String.substr_index detail ~pattern:from) ~f:(fun i ->
      let rest = String.drop_prefix detail (i + String.length from) in
      Option.map (String.substr_index rest ~pattern:until) ~f:(fun j -> String.prefix rest j))

(* A configuration-wide [-Werror] echoes the routine's declaration in BOTH the kernel's and the
   probe's compiler output; routine-name attribution must not take that for a name collision. The
   same report under [Compiler_bug] is attributed, which is what makes the report a discriminating
   one. *)
let attribution_keeps severity =
  let routine = "gh1350_attribution" in
  let detail =
    Printf.sprintf
      "kernel.c:3:6: error: no previous prototype for function '%s' [-Werror,-Wmissing-prototypes]\n\
      \    3 | void %s(void) {\n"
      routine routine
  in
  let exn =
    Outcome.Cause_at
      ( Outcome.Backend_compile,
        Outcome.Backend_rejected { backend = "cc"; stage = "test"; severity; detail } )
  in
  match
    Ir.C_syntax.attribute_routine_name_failure
      ~routine_names:[ (routine, routine) ]
      exn (Stdlib.Printexc.get_callstack 1)
  with
  | () -> false
  | exception raised -> phys_equal raised exn

let () =
  Stdio.printf "mode: %s\n" mode;
  p "a configuration fault echoing the routine's declaration is not taken for its name's failure"
    (attribution_keeps Outcome.Configuration);
  p "the same report under Compiler_bug is attributed to the routine name"
    (not (attribution_keeps Outcome.Compiler_bug));
  let ctx = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name ctx);
  let x = TDSL.range 16 in
  let y =
    if String.equal mode "macro_label" then
      (* The label the [macro_label] rule's compiler command defines as a macro. *)
      let%op gh1338_macro_node = (2. *. x) + 1. in
      gh1338_macro_node
    else
      let%op y = (2. *. x) + 1. in
      y
  in
  Train.set_materialized y.value;
  let comp = Train.forward y in
  let outcome =
    Context.compile_outcome ~name:"gh1350_attribution" ~provenance:Outcome.Candidate
      ~candidate:"attribution probe" ctx comp Ir.Indexing.Empty
  in
  let failure =
    match outcome with
    | Ok _ -> None
    | Error
        (Outcome.Fatal
           { cause = Some (Outcome.Backend_rejected { backend; stage; severity; detail }); _ }) ->
        Some (`Fatal, { backend; stage; severity; detail })
    | Error
        (Outcome.Classified
           { cause = Outcome.Backend_rejected { backend; stage; severity; detail }; _ }) ->
        Some (`Declined, { backend; stage; severity; detail })
    | Error _ -> None
  in
  Option.iter failure ~f:(fun (_, { detail; _ }) ->
      Stdio.eprintf "failure detail (not part of the golden):\n%s\n%!" detail);
  let has f = Option.exists failure ~f in
  let detail_has substring =
    has (fun (_, { detail; _ }) -> String.is_substring detail ~substring)
  in
  let claims_a_bug = detail_has Ir.C_syntax.codegen_bug_claim_prefix in
  let compile_exn =
    match Context.compile ctx comp Ir.Indexing.Empty with
    | _ -> None
    | exception exn ->
        Stdio.eprintf "Context.compile raised (not part of the golden): %s\n%!" (Exn.to_string exn);
        Some exn
  in
  p "the compile fails with a typed cc rejection"
    (has (fun (_, { backend; _ }) -> String.equal backend "cc"));
  match culprit with
  | Some setting ->
      p "the failure ends a search: it is fatal, not a candidate decline"
        (has (fun (fate, _) -> Poly.equal fate `Fatal));
      p "the stage is the toolchain's"
        (has (fun (_, { stage; _ }) -> String.equal stage Outcome.toolchain_stage));
      p "the severity is Configuration"
        (has (fun (_, { severity; _ }) -> Outcome.equal_severity severity Outcome.Configuration));
      p "the detail names the broken setting" (detail_has setting);
      p_none "the detail names no other setting"
        (List.filter settings ~f:(fun s -> not (String.equal s setting)))
        ~f:detail_has;
      p "the detail carries the probe's compiler output"
        (has (fun (_, { detail; _ }) ->
             Option.exists
               (section detail ~from:"Probe output:\n" ~until:"\nKernel compilation command:")
               ~f:(fun output -> not (String.is_empty (String.strip output)))));
      p "the detail does not call it a bug in OCANNL" (not claims_a_bug);
      p "Context.compile raises a user error naming the setting"
        (match compile_exn with
        | Some (Utils.User_error msg) -> String.is_substring msg ~substring:setting
        | _ -> false)
  | None ->
      p "the failure is the compiler stage's, as a codegen failure"
        (has (fun (_, { stage; _ }) -> String.equal stage "compiler"));
      p "the severity is Compiler_bug"
        (has (fun (_, { severity; _ }) -> Outcome.equal_severity severity Outcome.Compiler_bug));
      p "the detail keeps the bug claim" claims_a_bug;
      p "Context.compile raises Invalid_argument with the bug claim"
        (match compile_exn with
        | Some (Invalid_argument msg) ->
            String.is_substring msg ~substring:Ir.C_syntax.codegen_bug_claim_prefix
        | _ -> false)
