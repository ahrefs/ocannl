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
  Verdict.case "a named einsum form is its operator's spelling" (fun () ->
      (* Both einsum spellings leave the constrained operand's application responsible for its own
         site. A type annotation must not suppress that application's frame. *)
      let named_line = ref 0 and operator_line = ref 0 in
      let named =
        attempt "named form"
          (tensor (fun () ->
               named_line := Stdlib.__LINE__ + 1;
               [%op einsum1 "...|a => ...|a" (block x : Tensor.t)]))
      in
      let operator =
        attempt "operator"
          (tensor (fun () ->
               operator_line := Stdlib.__LINE__ + 1;
               [%op (block x : Tensor.t) ++ "...|a => ...|a"]))
      in
      p "both spellings are refused" (Option.is_some named && Option.is_some operator);
      p "the named form's constrained operand names its own application site"
        (names_site named ~expected:(in_op !named_line));
      p "the operator's constrained operand names its own application site"
        (names_site operator ~expected:(in_op !operator_line)));
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

(* gh-ocannl-2164: a contradiction independent of [x] fails while tracing at [()]. *)
let%op bad_definition ~width () x = x + ({ left = 0.; o = [ width ] } + { right = 0.; o = [ 4 ] })

let () =
  Tensor.unsafe_reinitialize ();
  Verdict.case "a trace failure at closure construction names the application" (fun () ->
      let line = ref 0 in
      let msg =
        attempt "trace in %op" (fun () ->
            line := Stdlib.__LINE__ + 1;
            ignore ([%op bad_definition ~width:3 ()] : Tensor.t -> Tensor.t))
      in
      p "the trace names the closure construction's file and line"
        (names_site msg ~expected:(in_op !line));
      p "the trace starts with its phase and function"
        (Option.exists msg ~f:(fun msg ->
             String.is_prefix msg ~prefix:"tracing the shape scheme bad_definition in "));
      let msg =
        attempt "trace in plain OCaml" (fun () ->
            ignore (bad_definition ~width:3 () : Tensor.t -> Tensor.t))
      in
      p "a plain trace after a refusal names no site"
        (Option.exists msg ~f:(fun msg ->
             String.is_prefix msg ~prefix:"tracing the shape scheme bad_definition: ")));
  Verdict.case "a definition check preserves the trace's application site" (fun () ->
      let msg, line =
        let line = ref 0 in
        let msg =
          attempt "definition check" (fun () ->
              ignore
                (Tensor.Scheme.check_definition ~name:"bad_definition" (fun () ->
                     line := Stdlib.__LINE__ + 1;
                     ignore ([%op bad_definition ~width:2 ()] : Tensor.t -> Tensor.t))))
        in
        (msg, !line)
      in
      p "the definition check names the closure construction's file and line"
        (names_site msg ~expected:(in_op line));
      p "the error retains the definition-check context"
        (Option.exists msg ~f:(fun msg ->
             String.is_prefix msg ~prefix:"checking the definition bad_definition: ")))

let%op bad_deferred x = x + ([ 1.; 2.; 3. ] + [ 1.; 2.; 3.; 4. ])

let () =
  Verdict.case "a deferred trace names its first application" (fun () ->
      let x = input "deferred_x" [ 3 ] in
      let line, msg =
        (Stdlib.__LINE__, attempt "deferred trace" (tensor (fun () -> [%op bad_deferred x])))
      in
      p "the deferred trace names the first application's file and line"
        (names_site msg ~expected:(in_op line)))

let%op _direct_add x y = x + y + ([ 1.; 2.; 3. ] + [ 1.; 2.; 3.; 4. ])

(* Pin both the final-call site (outer wrapper, after the partial call returns) and the
   closure-construction site (inner wrapper). Formatting is disabled so the pipeline stages keep
   distinct source lines. _direct_add traces only once both tensor arguments have arrived. *)
let () =
  Tensor.unsafe_reinitialize ();
  Verdict.case "ordinary partial pipelines record both application stages" (fun () ->
      let x = input "partial_x" [ 3 ] and y = input "partial_y" [ 3 ] in
      let direct_line = ref 0 and pipe_line = ref 0 and partial_line = ref 0 in
      let direct =
        attempt "direct partial trace"
          (tensor (fun () ->
               direct_line := Stdlib.__LINE__ + 1;
               [%op _direct_add y x]))
      in
      let piped =
        attempt "piped final trace"
          (tensor (fun () ->
               pipe_line := Stdlib.__LINE__ + 1;
               [%op x
                 |> _direct_add y]))
      in
      p "the direct call's trace names the full application"
        (names_site direct ~expected:(in_op !direct_line));
      p "the piped call's trace names the final application, after the partial wrapper returns"
        (names_site piped ~expected:(in_op !pipe_line));
      let partial =
        attempt "piped partial trace"
          (tensor (fun () ->
               partial_line := Stdlib.__LINE__ + 2;
               [%op x
                 |> bad_definition ~width:3 ()]))
      in
      p "a trace while constructing the piped closure names the partial application"
        (names_site partial ~expected:(in_op !partial_line)))
[@@ocamlformat "disable"]

(* A committed-state read of an older open tensor raises during tracing and makes rewind unsafe. *)
let%op rewind_reads_dims c () x =
  ignore (Shape.to_dims c.Tensor.shape : int array);
  x + c

let%op warning_sets_outside r () x =
  [%oc Shape.set_dim r 4];
  x + x

let () =
  Verdict.case "a refused trace rewind names its application" (fun () ->
      Tensor.unsafe_reinitialize ();
      let c = input "rewind_c" [ 3 ] in
      let line = ref 0 in
      let msg =
        attempt "rewind in %op" (fun () ->
            line := Stdlib.__LINE__ + 1;
            ignore ([%op rewind_reads_dims c ()] : Tensor.t -> Tensor.t))
      in
      p "the rewind refusal names the function followed by the application site"
        (names_site msg
           ~expected:
             ("the shape scheme rewind_reads_dims "
             ^ String.chop_suffix_exn (in_op !line) ~suffix:": "
             ^ " was refused"));
      p "the refusal keeps the opt-out remedy" (names_site msg ~expected:"[@@ocannl.no_scheme]");
      Tensor.unsafe_reinitialize ();
      let c = input "plain_rewind_c" [ 3 ] in
      let msg =
        attempt "rewind in plain OCaml" (fun () ->
            ignore (rewind_reads_dims c () : Tensor.t -> Tensor.t))
      in
      p "a plain rewind refusal after the located failure has no site"
        (names_site msg ~expected:"the shape scheme rewind_reads_dims was refused"
        && not (names_site msg ~expected:"in the %op application at")));
  Verdict.case "ordinary fallback warnings stay independent of the application site" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "warning_dim" in
      let (), warnings =
        Stderr_capture.capture (fun () ->
            ignore ([%op warning_sets_outside r ()] : Tensor.t -> Tensor.t);
            ignore ([%op warning_sets_outside r ()] : Tensor.t -> Tensor.t))
      in
      let warnings = String.split_lines warnings in
      p "one fallback warning is printed for both application sites"
        (Int.equal 1
           (List.count warnings ~f:(fun s ->
                String.is_substring s
                  ~substring:"the shape scheme of warning_sets_outside was refused")));
      p_none "fallback warnings contain no application site" warnings ~f:(fun s ->
          String.is_substring s ~substring:"in the %op application at"))
