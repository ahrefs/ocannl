open Base
open Ocannl
open Nn_blocks.DSL_modules

(* Direct operation calls still need Shape's runtime fallback for absent captures. Literal operator
   captures are now refused by the ppx (test/ppx/test_capture_errors.ml, gh-ocannl-2163). The
   runtime hint uses the parsed labels because callers can supply comma-rendered specs. *)

let capture spec label x = NTDSL.einsum1 ~capture_dims:[ Shape.get_variable_ref label ] spec x ()
let input () = NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 3 ] ~f:(fun _ -> 1.) ()

let refusal name build =
  Tensor.unsafe_reinitialize ();
  match build (input ()) with
  | (_ : Tensor.t) -> Stdio.printf "%s: no refusal\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s:\n  %s\n" name msg

let () =
  refusal "multi-letter label, single-char spec" (fun x -> capture "ab => a" "ab" x);
  refusal "label with a digit, single-char spec" (fun x -> capture "f0 => f0" "f0" x);
  refusal "single-char spec through the operation, not the ppx" (fun x ->
      Operation.einsum1
        ~capture_dims:[ Shape.get_variable_ref "ab" ]
        "ab => a" ~grad_spec:Tensor.Prohibit_grad x ());
  (* A comma-rendered spec still receives the conservative runtime hint. *)
  refusal "multi-letter label whose letters a multichar spec names" (fun x ->
      capture "a, b => a" "ab" x);
  refusal "one-letter label absent from the spec" (fun x -> capture "ab => a" "c" x);
  refusal "multi-letter label with a letter the spec does not hold" (fun x ->
      capture "ab => a" "bc" x);
  refusal "label with a digit the spec does not hold" (fun x -> capture "ab => a" "a1" x);
  (* The fix the hint names: the label written as one multichar axis is found and captures it. *)
  Tensor.unsafe_reinitialize ();
  let%op y = input () ++ "a, f0 => a" [ "f0" ] in
  ignore (Shape.to_dims y.Tensor.shape : int array);
  Stdio.printf "multichar spec: the label captures dimension %s\n"
    (Option.value_map f0.var_ref.solved_dim ~default:"none" ~f:Int.to_string)
