(** The claim helpers of the lineage-refusal tests (gh-ocannl-1724, gh-ocannl-1744, gh-ocannl-1781):
    a refusal's text, an accepted call, an exact readback, a tensor's gradient node and a substring
    test. Six tests carried their own copies, one of them without the stderr echo of the refusal.

    Call them qualified, through an alias ([module LC = Lineage_claims], then [LC.refusal ...]),
    never under an [open]: past an open of a module [Verdict_callee] cannot read, a test's
    unqualified [p] reaches [Verdict] only ambiguously. *)

open Base
open Ocannl
open Verdict.Claims

(** The gradient node of a differentiable tensor; raises on a tensor without one. *)
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(** Whether [msg] contains [substring]. *)
let has msg substring = String.is_substring msg ~substring

(** A readback equal to [want] cell for cell. *)
let exact name got want = p_close name got want ~tol:(fun _ -> 0.)

(** The text of the [User_error] [f] raises, echoed to stderr outside the golden; a failed claim
    when it raises none. *)
let refusal name f =
  match f () with
  | _ ->
      fail (name ^ ": was accepted");
      ""
  | exception Utils.User_error msg ->
      Stdio.eprintf "%s: %s (not part of the golden)\n%!" name msg;
      msg

(** [Some] of what [f] returns, with a passing claim; a failed claim naming the refusal, and [None],
    when it raises a [User_error]. *)
let accepted name f =
  match f () with
  | x ->
      p name true;
      Some x
  | exception Utils.User_error msg ->
      p (name ^ " (refused: " ^ msg ^ ")") false;
      None
