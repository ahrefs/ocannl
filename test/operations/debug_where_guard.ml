(* task-9658aac9: a debug value-logged [Where] must not emit its conditionally-evaluated branch
   reads as unconditional printf accessor arguments -- the dereference must short-circuit on the
   (negated) condition, exactly as the runtime ternary does. This is the directly reachable form of
   the Stage B unit-solve guard hazard: that guard's then-branch is a producer [Get] at a unit-solved
   index that can be out of bounds for non-matching kept-loop iterations, but in the current
   virtualized flow it lives inside a non-logged [Local_scope] ([log_set_locals:false]); a top-level
   [where] op exercises the same [debug_float] [Where] arm directly. Mirrors the gh-343
   [Get_dynamic] remedy (commit 18964416): we read the generated debug-logging C and assert the
   branch reads are gated by a ternary, not emitted as bare unconditional arguments.

   We need [debug_log_from_routines] (gated on [log_level > 1]), which makes config retrieval verbose
   and backend-specific; to keep stdout clean and backend-independent we suppress stdout (the [.c]
   file is written to disk regardless) around the compile and print only the boolean assertions. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

let p name b = Stdio.printf "%s: %b\n" name b

(* Route stdout to /dev/null around [f]; the generated [build_files/*.c] is written to disk and is
   unaffected. Keeps the deterministic test output free of (backend-specific) config-retrieval noise
   that [log_level > 1] turns on. *)
let with_stdout_to_devnull f =
  Stdlib.flush Stdlib.stdout;
  let saved = Unix.dup Unix.stdout in
  let dn = Unix.openfile Stdlib.Filename.null [ Unix.O_WRONLY ] 0o600 in
  Unix.dup2 dn Unix.stdout;
  Unix.close dn;
  Exn.protect ~f ~finally:(fun () ->
      Stdlib.flush Stdlib.stdout;
      Unix.dup2 saved Unix.stdout;
      Unix.close saved)

let read_all_generated_c () =
  let dir = "build_files" in
  if Stdlib.Sys.file_exists dir then
    Stdlib.Sys.readdir dir |> Array.to_list
    |> List.filter ~f:(fun f -> String.is_suffix f ~suffix:".c")
    |> List.sort ~compare:String.compare
    |> List.map ~f:(fun f -> Stdio.In_channel.read_all (Stdlib.Filename.concat dir f))
    |> String.concat ~sep:"\n"
  else ""

let () =
  Tensor.unsafe_reinitialize ();
  Utils.set_log_level 2;
  Utils.settings.debug_log_from_routines <- true;
  Utils.settings.output_debug_files_in_build_directory <- true;
  with_stdout_to_devnull (fun () ->
      let ctx = Context.auto () in
      let cond = Tensor.number ~label:[ "cond" ] 0.0 in
      let a = Tensor.number ~label:[ "athen" ] 2.0 in
      let b = Tensor.number ~label:[ "belse" ] 3.0 in
      let result = Operation.where ~grad_spec:Prohibit_grad cond a b () in
      Train.set_materialized cond.value;
      Train.set_materialized a.value;
      Train.set_materialized b.value;
      Train.set_materialized result.value;
      let _ctx = Train.forward_once ctx result in
      ());
  let c = read_all_generated_c () in
  if String.is_empty c then
    (* Non-C-family backend (e.g. metal writes .metal, cuda .cu): the C assertion does not apply. *)
    p "debug Where branch reads are short-circuited (skipped: non-C backend)" true
  else begin
    (* The debug value-log renders the Where value via [debug_float]. After the fix, each branch's
       dereferencing printf argument is gated on the (negated) condition:
         then -> (cond ? athen[..] : 0)        else -> (!(cond) ? belse[..] : 0)
       The " : 0)" suffix and the "(!(" negation are produced only by the guard wrapper -- the
       runtime ternary's else is the raw branch read, never a literal 0 -- so their presence is a
       precise, name-independent signal that both branch reads short-circuit. Before the fix the
       branch reads were emitted as bare unconditional printf arguments (no " : 0)", no "(!("). *)
    p "debug Where then-branch read is gated by the condition"
      (String.is_substring c ~substring:" ? " && String.is_substring c ~substring:" : 0)");
    p "debug Where else-branch read is gated by the negated condition"
      (String.is_substring c ~substring:"(!(");
    (* AC2 fidelity: the annotated display still shows both branch sub-values (the [debug_float]
       display doc is unchanged; only the printf argument expressions are guarded). *)
    p "debug Where still logs both branch sub-values (annotated display preserved)"
      (String.is_substring c ~substring:"!= 0.0 ?")
  end
