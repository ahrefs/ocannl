open Base
open Ocannl
open Nn_blocks.DSL_modules

(* A capture label is looked up whole among the spec's labels, but a single-char spec lexes a
   multi-letter label as one axis per character, so the label is never there (gh-ocannl-2092). The
   refusal then names the fix -- multichar mode -- with the note the einsum parser gives a
   multi-letter row variable. The ppx renders every literal spec comma-separated ("ab => a" reaches
   Shape as "a, b => a"), so the hint is decided on the spec's labels: it is given when every
   character of the label is an axis of the spec, and withheld for a one-letter label or a label
   with a character the spec does not hold. *)

let input () = NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 3 ] ~f:(fun _ -> 1.) ()

let refusal name build =
  Tensor.unsafe_reinitialize ();
  match build (input ()) with
  | (_ : Tensor.t) -> Stdio.printf "%s: no refusal\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s:\n  %s\n" name msg

let () =
  refusal "multi-letter label, single-char spec" (fun x ->
      let%op y = x ++ "ab => a" [ "ab" ] in
      ignore (ab : Shape.delayed_var_ref);
      y);
  refusal "label with a digit, single-char spec" (fun x ->
      let%op y = x ++ "f0 => f0" [ "f0" ] in
      ignore (f0 : Shape.delayed_var_ref);
      y);
  refusal "single-char spec through the operation, not the ppx" (fun x ->
      Operation.einsum1
        ~capture_dims:[ Shape.get_variable_ref "ab" ]
        "ab => a" ~grad_spec:Tensor.Prohibit_grad x ());
  (* Indistinguishable, once the ppx has rendered it, from the single-char case above. *)
  refusal "multi-letter label whose letters a multichar spec names" (fun x ->
      let%op y = x ++ "a, b => a" [ "ab" ] in
      ignore (ab : Shape.delayed_var_ref);
      y);
  refusal "one-letter label absent from the spec" (fun x ->
      let%op y = x ++ "ab => a" [ "c" ] in
      ignore (c : Shape.delayed_var_ref);
      y);
  refusal "multi-letter label with a letter the spec does not hold" (fun x ->
      let%op y = x ++ "ab => a" [ "bc" ] in
      ignore (bc : Shape.delayed_var_ref);
      y);
  refusal "label with a digit the spec does not hold" (fun x ->
      let%op y = x ++ "ab => a" [ "a1" ] in
      ignore (a1 : Shape.delayed_var_ref);
      y);
  (* The fix the hint names: the label written as one multichar axis is found and captures it. *)
  Tensor.unsafe_reinitialize ();
  let%op y = input () ++ "a, f0 => a" [ "f0" ] in
  let ctx = Train.forward_once (Context.auto ()) y in
  Stdio.printf "multichar spec: the label captures dimension %s\n"
    (Option.value_map f0.var_ref.solved_dim ~default:"none" ~f:Int.to_string);
  Context.release ctx
