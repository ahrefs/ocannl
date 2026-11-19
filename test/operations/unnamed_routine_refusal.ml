(* A compile given a nameless comp names the fix in its refusal's first line.

   [Context.compile ctx t.Tensor.forward] -- the bare forward code, with no [Train.forward] around
   it -- has no [Assignments.Block_comment] to derive the routine's name from, so lowering refuses
   it through [Assignments.get_name_exn]. That refusal used to read "no comments in code: " followed
   by the whole code, and said nothing of the fix. Its first line now says the routine has no name
   and names both remedies, [Train.forward t] and the compile's [?name]; the code follows, cut to
   its head. The claims: the refusal stays an [Invalid_argument] (the readers that fall back on it,
   such as [Context]'s ["(unnamed)"] label for its shared-gradient refusal, match that constructor),
   its first line is the golden's, the comp's own code follows that line, a long comp's code is cut
   to its first lines with a count of the rest, and both remedies the line names compile the same
   tensor. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
open Verdict.Claims

let first_line msg = Option.value (List.hd (String.split_lines msg)) ~default:""

(* The comp's code as the refusal renders it, so the claims pin what follows the first line to the
   comp itself, whatever the identifier style. *)
let rendering (comp : Asgns.comp) = String.split_lines (Asgns.to_string comp.Asgns.asgns)
let cut = 20

let () =
  let x = TDSL.range 5 ~label:[ "unr_x" ] in
  let%op y = (x *. 2.) + 1. in
  let ctx = Context.auto () in
  let refusal =
    match Context.compile ctx y.Tensor.forward Ir.Indexing.Empty with
    | (_ : Context.t * Context.routine) -> None
    | exception Invalid_argument msg -> Some msg
  in
  p "compiling a bare t.forward is refused with an Invalid_argument" (Option.is_some refusal);
  let msg = Option.value refusal ~default:"" in
  Stdio.printf "first line: %s\n" (first_line msg);
  let code = match String.split_lines msg with _ :: "The code:" :: code -> code | _ -> [] in
  p_all2 "the comp's own code follows the first line, whole" (Array.of_list code)
    (Array.of_list (rendering y.Tensor.forward))
    ~f:String.equal;
  (* A comp long enough to be cut: a chain of [steps] additions. *)
  let steps = 40 in
  let long =
    Fn.apply_n_times ~n:steps
      (fun t ->
        let%op t' = t + 1. in
        t')
      (TDSL.range 3 ~label:[ "unr_l" ])
  in
  let long_msg =
    match Asgns.get_name_exn long.Tensor.forward.Asgns.asgns with
    | name -> "derived " ^ name
    | exception Invalid_argument msg -> msg
  in
  let whole = rendering long.Tensor.forward in
  Stdio.eprintf "long comp: %d lines of code (not part of the golden)\n%!" (List.length whole);
  p "the long comp's code is longer than the cut" (List.length whole > cut);
  let head, rest = List.split_n whole cut in
  p_all2 "a long comp's code is cut to its head, saying how much was left out"
    (Array.of_list (String.split_lines long_msg))
    (Array.of_list
       ((first_line msg :: "The code:" :: head)
       @ [ Printf.sprintf "... (%d more lines)" (List.length rest) ]))
    ~f:String.equal;
  (* Both remedies the first line names. *)
  let named = TDSL.range 5 ~label:[ "unr_named_x" ] in
  let%op named = (named *. 2.) + 1. in
  p "the compile's ~name names the bare forward"
    (match Context.compile ~name:"unr_named" ctx named.Tensor.forward Ir.Indexing.Empty with
    | (_ : Context.t * Context.routine) -> true
    | exception exn ->
        Stdio.eprintf "~name: %s (not part of the golden)\n%!" (Exn.to_string exn);
        false);
  p "Train.forward names the refused tensor's own forward"
    (match Context.compile ctx (Train.forward y) Ir.Indexing.Empty with
    | (_ : Context.t * Context.routine) -> true
    | exception exn ->
        Stdio.eprintf "Train.forward: %s (not part of the golden)\n%!" (Exn.to_string exn);
        false)
