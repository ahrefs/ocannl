open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* Shape refusals raised through [%op]/[%cd] quote the einsum spec the user wrote (gh-ocannl-2163).
   A literal reaches Shape as written; a spec reading a runtime value (a stride identifier, a
   padding-less [+] window) is built at run time and carries its source for the messages. Each case
   claims the message cites the source and not the comma-separated re-rendering the translators used
   to pass on. Direct operation calls and computed specs have no source: they quote what they
   pass. *)

let input ?(l = "x") dims = NTDSL.init ~l ~prec:Ir.Ops.single ~b:[] ~o:dims ~f:(fun _ -> 1.) ()

let refusal ?absent name ~cites build =
  Tensor.unsafe_reinitialize ();
  match build () with
  | () -> p (name ^ ": refused") false
  | exception Row.Shape_error (msg, _) ->
      Stdio.printf "%s:\n  %s\n" name msg;
      p (name ^ ": cites the source spec") (String.is_substring msg ~substring:cites);
      Option.iter absent ~f:(fun absent ->
          p
            (name ^ ": does not cite the re-rendered spec")
            (not (String.is_substring msg ~substring:absent)))

let forward (t : Tensor.t) = ignore (t.Tensor.forward : Ir.Assignments.comp)
let%cd cd_concat x = (x ++ "ab => a^b").forward

let () =
  (* Literal specs: the re-rendering was multichar with commas for a single-char spec, and spaced
     its concatenations. *)
  refusal "%op unary, arity" ~cites:"spec (expected one argument): ab; c => a"
    ~absent:"a, b; c => a" (fun () ->
      let x = input [ 2; 3 ] in
      let%op r = x ++ "ab; c => a" in
      forward r);
  refusal "%op binary, a result concatenation no operand names"
    ~cites:"einsum spec \"ab; c => a^b\"" ~absent:"a ^ b" (fun () ->
      let x = input [ 2; 3 ] in
      let y = input ~l:"y" [ 4 ] in
      let%op r = x +* "ab; c => a^b" y in
      forward r);
  refusal "%op block, a result concatenation no operand names" ~cites:"einsum spec \"ab; c => a^b\""
    ~absent:"a ^ b" (fun () ->
      let x = input [ 2; 3 ] in
      let y = input ~l:"y" [ 4 ] in
      let%op r = (x, y) ++^ "ab; c => a^b" in
      forward r);
  refusal "%cd unary, a result concatenation no operand names" ~cites:"einsum spec \"ab => a^b\""
    ~absent:"a ^ b" (fun () -> ignore (cd_concat (input [ 2; 3 ]) : Ir.Assignments.comp));
  (* A refusal raised at closure, from a fixed-index site the solver stored. *)
  refusal "%op fixed index into a parameter, at closure" ~cites:"einsum spec \"...|hw => ...|00\""
    ~absent:"h, w" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3; 3 ] () in
      let%op pooled = x ++ "...|hw => ...|00" in
      let%op logits = ({ w } * pooled) + { b = 0.; o = [ 5 ] } in
      ignore (Train.init_params (Context.auto ()) Train.IDX.empty logits : Context.t));
  (* Runtime specs: Shape parses the evaluated spec and quotes the source; the axis in the detail is
     the evaluated one. *)
  refusal "%op binary, a stride identifier at 0"
    ~cites:"affine axis `0*o<+k` in einsum spec \"stride*o<+k; k => o\"" ~absent:"spec \"0*o"
    (fun () ->
      let stride = 0 in
      let x = input [ 5 ] in
      let w = input ~l:"w" [ 3 ] in
      let%op r = x +* "stride*o<+k; k => o" w in
      forward r);
  refusal "%op binary, a padding-less window reading use_padding"
    ~cites:"result-side padded window `i=+k` in einsum spec \"s, i; k => s, i+k\""
    ~absent:"spec \"s, i; k => s, i=+k" (fun () ->
      let use_padding = true in
      let x = input [ 2; 5 ] in
      let w = input ~l:"w" [ 3 ] in
      let%op r = x +* "s, i; k => s, i+k" w in
      forward r);
  refusal "%op block, a stride identifier at 0"
    ~cites:"affine axis `0*a` in einsum spec \"stride*a; b => a^b\"" ~absent:"spec \"0*a" (fun () ->
      let stride = 0 in
      let x = input [ 2 ] in
      let y = input ~l:"y" [ 3 ] in
      let%op r = (x, y) ++^ "stride*a; b => a^b" in
      forward r);
  refusal "%op named einsum1, a stride identifier at 0"
    ~cites:"affine axis `0*o` in einsum spec \"stride*o => o\"" ~absent:"spec \"0*o" (fun () ->
      let stride = 0 in
      let x = input [ 5 ] in
      let%op r = einsum1 "stride*o => o" x in
      forward r);
  refusal "%op named einsum3, a stride identifier at 0"
    ~cites:"affine axis `0*i` in einsum spec \"stride*i; i; i => i\"" ~absent:"spec \"0*i"
    (fun () ->
      let stride = 0 in
      let x = input [ 5 ] in
      let y = input ~l:"y" [ 5 ] in
      let z = input ~l:"z" [ 5 ] in
      let%op r = einsum3 "stride*i; i; i => i" x y z in
      forward r);
  (* No ppx provenance: a direct call and a computed spec quote the string they pass. *)
  refusal "direct Operation call" ~cites:"spec (expected one argument): ab; c => a" (fun () ->
      forward (NTDSL.einsum1 "ab; c => a" (input [ 2; 3 ]) ()));
  refusal "computed spec" ~cites:"spec (expected one argument): a, b; c => a" (fun () ->
      let spec = String.concat [ "a, b"; "; c => a" ] in
      let x = input [ 2; 3 ] in
      let%op r = x ++ spec in
      forward r)
