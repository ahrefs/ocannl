open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Surface-level probes for known/expected incompleteness. These are not regressions to preserve
   forever; if inference grows powerful enough for the "abstract param shape" case to succeed, the
   corresponding report section should be updated and this probe can be relaxed or removed. *)

let failures = ref 0

let fail msg =
  Int.incr failures;
  Stdio.printf "UNEXPECTED: %s\n" msg

let run name ~expect f =
  Tensor.unsafe_reinitialize ();
  Shape.unsafe_reinitialize ();
  Stdio.printf "=== %s ===\n" name;
  match
    try
      let ctx = Context.auto () in
      let y = f () in
      let _ctx = Train.forward_once ctx y in
      `Success (Shape.to_string_hum y.shape)
    with Row.Shape_error (msg, _) -> `Shape_error msg
  with
  | `Success shape -> (
      Stdio.printf "no error: %s\n" shape;
      match expect with `Success -> () | `Shape_error -> fail "expected Shape_error")
  | `Shape_error msg -> (
      Stdio.printf "Shape_error: %s\n" msg;
      match expect with `Shape_error -> () | `Success -> fail "expected success")

let () =
  run "cyclic row permute added to original, concrete periodic shape" ~expect:`Success (fun () ->
      let%op x = { x = uniform1 (); o = [ 3; 5; 3 ] } in
      let%op y = x + (x ++ "a,b,..r.. => ..r..,b,a" [ "a"; "b"; "r" ]) in
      y)

let () =
  run "cyclic row permute added to original, abstract param shape" ~expect:`Shape_error (fun () ->
      let%op x = { x = uniform1 () } in
      let%op y = x + (x ++ "a,b,..r.. => ..r..,b,a" [ "a"; "b"; "r" ]) in
      Shape.set_dim a 3;
      Shape.set_dim b 5;
      y)

let () = if !failures > 0 then Stdlib.exit 1
