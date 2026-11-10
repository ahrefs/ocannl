(* gh-ocannl-2007: a shape-scheme application error names where the application is. The [%op]
   expansion runs each function application written in [%op] code with its file and line recorded,
   and a check raised inside one reads "in the %op application at <file>:<line>"; an explicit
   [Tensor.Scheme.instantiate ~here:[%here]] reads "at <file>:<line>". An application written in
   plain OCaml, outside any [%op] application, has no site: OCaml gives a function no caller
   location. Each claim derives the expected line from [__LINE__] or [[%here]] at the application,
   so it pins the relationship; the golden prints the messages themselves. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let input ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

let site line = Printf.sprintf "%s:%d" Stdlib.__FILE__ line
let in_op line = "in the %op application at " ^ site line ^ ": "
let%op mlp_layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })

let%op residual_block ~hid () =
  let layer = mlp_layer ~hid () in
  fun x -> x + layer x

(* Plain OCaml: an application it makes is inside the [%op] application that called it. *)
let apply_plain f x = f x

(* Not schemed (its labelled parameter is not a tensor): the application in its body is checked
   itself, naming its own line. *)
let%op through ~layer x = layer x
let through_line = Stdlib.__LINE__ - 1

(* The headline of a refusal, printed; [None] when the application succeeds. *)
let attempt name f =
  match f () with
  | () ->
      Stdio.printf "%s: ok\n" name;
      None
  | exception Row.Shape_error (msg, _) ->
      Stdio.printf "%s:\n  %s\n" name msg;
      Some msg

let tensor f () = ignore (f () : Tensor.t)

let names_site msg ~expected =
  Option.exists msg ~f:(fun msg -> String.is_substring msg ~substring:expected)

let names_no_site msg =
  Option.exists msg ~f:(fun msg ->
      String.is_prefix msg ~prefix:"applying the shape scheme"
      && (not (String.is_substring msg ~substring:"in the %op application at"))
      && not (String.is_substring msg ~substring:(Stdlib.__FILE__ ^ ":")))

let () =
  Tensor.unsafe_reinitialize ();
  let x = input ~batch:[ 2 ] "x" [ 3 ] in
  let block = residual_block ~hid:4 () in
  Verdict.case "an application in %op code names its line" (fun () ->
      let line, msg = (Stdlib.__LINE__, attempt "%op" (tensor (fun () -> [%op block x]))) in
      p "the check names the application's file and line" (names_site msg ~expected:(in_op line));
      (* The whole rendering: the origin of the scheme's equations names the site too. *)
      match [%op block x] with
      | _ -> p "the application is refused again" false
      | exception (Row.Shape_error _ as exn) ->
          Stdio.printf "%%op, rendered:\n%s\n" (Stdlib.Printexc.to_string exn));
  Verdict.case "an application in plain OCaml has no site" (fun () ->
      (* After a refused [%op] application: the site was put back when it raised. *)
      let msg = attempt "plain after a refusal" (tensor (fun () -> block x)) in
      p "a plain application's check names no site" (names_no_site msg);
      let fits = residual_block ~hid:3 () in
      ignore (attempt "%op, fitting" (tensor (fun () -> [%op fits x])) : string option);
      (* After an [%op] application that returned: the site was put back. *)
      let msg = attempt "plain after a success" (tensor (fun () -> block x)) in
      p "a plain application after a returned one names no site" (names_no_site msg));
  Verdict.case "an application reached from %op code is in that application" (fun () ->
      let line, msg =
        (Stdlib.__LINE__, attempt "via plain" (tensor (fun () -> [%op apply_plain block x])))
      in
      p "a plain function's application names the %op application that called it"
        (names_site msg ~expected:(in_op line));
      let line, msg = (Stdlib.__LINE__, attempt "pipe" (tensor (fun () -> [%op x |> block]))) in
      p "a pipe names its line" (names_site msg ~expected:(in_op line)));
  Verdict.case "an application in an unschemed function's body names that line" (fun () ->
      let layer = mlp_layer ~hid:4 () in
      ignore (attempt "first application, 3 features" (tensor (fun () -> layer x)) : string option);
      let x6 = input ~batch:[ 2 ] "x6" [ 6 ] in
      let msg = attempt "through" (tensor (fun () -> through ~layer x6)) in
      p "the body's application names its own line, not the caller's"
        (names_site msg ~expected:(in_op through_line)));
  Verdict.case "an explicit instance names [~here]" (fun () ->
      let needs3 = function [ x ] -> [ TDSL.O.(x + input "c3" [ 3 ]) ] | _ -> assert false in
      let s = Tensor.Scheme.trace ~label:"needs3" 1 needs3 in
      let x4 = input "x4" [ 4 ] in
      let result () = Shape.make ~debug_name:"r" ~id:(-1) () in
      let here = [%here] in
      let msg =
        attempt "instantiate ~here" (fun () ->
            Tensor.Scheme.instantiate ~here s ~args:[ x4.shape ] ~results:[ result () ])
      in
      p "the instance names [here]'s file and line"
        (names_site msg ~expected:("at " ^ site here.pos_lnum ^ ": "));
      (* The refused instance tainted the session, and [x4] is discarded with the open work. *)
      Tensor.rewind_to_last_closure ();
      let x4 = input "x4" [ 4 ] in
      let msg =
        attempt "instantiate" (fun () ->
            Tensor.Scheme.instantiate s ~args:[ x4.shape ] ~results:[ result () ])
      in
      p "without [~here], the instance names no site" (names_no_site msg))
