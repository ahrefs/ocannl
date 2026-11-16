open! Base
(** Two caught solver failures must retain their own diagnostics (gh-ocannl-2211). No kernel runs.
    Print the second exception, then revisit the first after the later solve. *)

open Ocannl.Operation.DSL_modules
open Verdict.Claims

let caught_conflict name a b =
  Tensor.unsafe_reinitialize ();
  let input suffix size =
    Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ^ suffix ] ~output_dims:[ size ] ()
  in
  let left = input "_left" a in
  let right = input "_right" b in
  match NTDSL.add left right () with
  | _ ->
      Verdict.fail "incompatible dimensions unexpectedly solved";
      assert false
  | exception (Row.Shape_error _ as exn) -> exn

let human = function
  | Row.Shape_error (msg, trace) -> Shape.error_to_string_hum msg trace
  | _ ->
      Verdict.fail "fixture caught an exception other than Shape_error";
      assert false

let renderers =
  [
    ("sexp_of_exn", fun exn -> Sexp.to_string_hum (sexp_of_exn exn));
    ("Exn.to_string", Exn.to_string);
    ("Printexc.to_string", Stdlib.Printexc.to_string);
  ]

let () =
  let first = caught_conflict "first_attempt" 2 3 in
  let before = List.map renderers ~f:(fun (_, render) -> render first) in
  let second = caught_conflict "second_attempt" 7 11 in
  p "second payload names its own solve"
    (String.is_substring (human second) ~substring:"second_attempt"
    && not (String.is_substring (human second) ~substring:"first_attempt"));
  List.iter renderers ~f:(fun (name, render) ->
      let text = render second in
      Stdio.printf "%s, second exception:\n%s\n" name text;
      p
        (name ^ " prints the second payload")
        (String.is_substring text ~substring:"second_attempt"
        && String.is_substring text ~substring:"7 vs 11"
        && not (String.is_substring text ~substring:"first_attempt")));
  List.iter2_exn renderers before ~f:(fun (name, render) before ->
      p
        (name ^ " preserves the first exception after a later solve")
        (String.equal before (render first)
        && String.is_substring before ~substring:"first_attempt"
        && String.is_substring before ~substring:"2 vs 3"))
