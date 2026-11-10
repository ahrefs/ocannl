(* gh-ocannl-1751: [%op] functions get their shape scheme automatically, in checking mode (pinned
   below against an ambient op_shape_schemes=off). The closure built at a function's [()]
   application traces the scheme of the function it returns (a function without a [()] stage: its
   first application); every application checks the scheme against its arguments' shapes, in a
   rewound window, before the body runs and registers as it always did. The oracle for "as it always
   did" is the same function opted out with [[@@ocannl.no_scheme]], or the whole run with the
   schemes switched off; each claim about the mechanism is paired with such a control, which must
   come out the other way. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let outcome f =
  match f () with
  | (_ : 'a) -> Ok ()
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg

let error_message f = match outcome f with Ok () -> None | Error msg -> Some msg

let has_substring ~substring = function
  | Some msg -> String.is_substring msg ~substring
  | None -> false

let leaf ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

(* The automatic schemes' events while [f] runs, oldest first. *)
let events f =
  let log = ref [] in
  let result = Tensor.For_testing.observe_schemes f ~on_event:(fun e -> log := e :: !log) in
  (result, List.rev !log)

let traced name = function
  | Tensor.Scheme.Traced s -> String.equal (Tensor.Scheme.label s) name
  | _ -> false

let checked name = function Tensor.Scheme.Checked n -> String.equal n name | _ -> false

let about name = function
  | Tensor.Scheme.Traced s -> String.equal (Tensor.Scheme.label s) name
  | Checked n | Unschemed (n, _) -> String.equal n name

(* This test is about checking mode. *)
let check_mode = Some `Check
let () = Tensor.For_testing.set_op_shape_schemes check_mode

(* {2 The functions under test} *)

let runs = ref 0

(* A layer with a lifted shared parameter: [w]'s output axis is given, its input axis comes from the
   first application. *)
let%op layer () x =
  Int.incr runs;
  ({ w; o = [ 4 ] } * x) + 1.0

let%op layer_plain () x =
  Int.incr runs;
  ({ w; o = [ 4 ] } * x) + 1.0
[@@ocannl.no_scheme]

(* A contradiction that does not depend on the argument. *)
let%op broken () x = x + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })

let%op broken_plain () x = x + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })
[@@ocannl.no_scheme]

(* Parameters after [()] that are not tensors, or not provably. *)
let%op with_step () ~train_step x = match train_step with Some () -> x + 1.0 | None -> x
let%op with_mask () ?mask x = match mask with None -> x | Some m -> x *. m

let%op rebound () ~s x =
  let s = s + 1.0 in
  s *. x

(* Labelled tensors, proven by their operand use. *)
let%op affine () ~gamma ~beta x = (gamma *. x) + beta

(* A returned function after the sub-layers are built: traced once they exist. *)
let%op stack () =
  let l1 = layer () in
  let l2 = layer () in
  fun x -> l2 (l1 x)

(* A closure built inside the body: inside the outer trace, it stays unschemed. *)
let%op const_layer () x = { cw = 0.5; o = [ 4 ] } * x

let%op outer () x =
  let inner = const_layer () in
  inner x

(* gh-ocannl-1814: who checks a nested application. [stack] above is the checked outer function. *)
let%op stack_plain () =
  let l1 = layer () in
  let l2 = layer () in
  fun x -> l2 (l1 x)
[@@ocannl.no_scheme]

(* A checked function reaching a schemed layer through an opted-out one. *)
let%op through_plain () =
  let middle = stack_plain () in
  fun x -> middle x

(* The outer trace is refused: its applications check nothing, so [layer]'s checks stay. *)
let%op refused_outer r () =
  let l = layer () in
  fun x ->
    [%oc Shape.set_dim r 4];
    l x

(* Without a [()] stage: traced at its first application, but not when that one is entailed. *)
let%op nested_late x = x + x
let%op over_late () x = nested_late x

(* Without a [()] stage, a contradiction in the pattern the outer function does not take: only this
   function's own first trace can find it. *)
let%op twice_when_masked ?(mask : Tensor.t option) x =
  match mask with
  | None -> x + x
  | Some m ->
      let y = (x *. m) ++ "...|a => ...|a" [ "a" ] in
      [%oc Shape.set_dim a 3];
      [%oc Shape.set_dim a 4];
      y

let%op over_twice_when_masked () x = twice_when_masked x

(* A body that raises once its check has passed. *)
let raise_in_body = ref false

let%op raising_outer () =
  let l = layer () in
  fun x ->
    let y = l x in
    [%oc if !raise_in_body then failwith "the body raised"];
    y

(* A closure built in a body and never applied: no window but its own holds its contradiction. *)
let%op builds_broken () x =
  let (_ : Tensor.t -> Tensor.t) = broken () in
  x

(* Random number generation (gh-ocannl-1890): a scheme carries a draw with its coefficient as a
   hole, which the conversion node each application's body registers fills. *)
let%op noisy () x = x + uniform ()

(* A draw whose precision comes from its call site: the pointwise product takes [x]'s. *)
let%op drawn () x = x *. uniform ()
let drawn_plain x = TDSL.O.(x *. uniform ())

(* A draw beside a tensor made before the closure: its precision chain reaches [c]. *)
let%op drawn_with c () x = x + (c *. uniform ())

(* A body setting a capture reference made outside it: a trace a scheme cannot carry. *)
let%op sets_outside r () x =
  [%oc Shape.set_dim r 4];
  x + x

(* A capture reference made outside the function (gh-ocannl-1791), by its caller: an input of the
   scheme, bound per application -- the sum over [x]'s last axis, whose size [r] captures. *)
let summed_runs = ref 0

let%op summed r () x =
  Int.incr summed_runs;
  [%oc NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" x ()]

let%op summed_plain r () x =
  Int.incr summed_runs;
  [%oc NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" x ()]
[@@ocannl.no_scheme]

(* Refusals the warning reports, each used by the warning's case alone: the warning is printed once
   per function and reason in a process. *)
let%op sets_warned r () x =
  [%oc Shape.set_dim r 4];
  x + x

let%op sets_quiet r () x =
  [%oc Shape.set_dim r 4];
  x + x
[@@ocannl.no_scheme]

let%op sets_off r () x =
  [%oc Shape.set_dim r 4];
  x + x

let%op const_inner () x = { cw2 = 0.5; o = [ 4 ] } * x

let%op outer_quiet () x =
  let inner = const_inner () in
  inner x

(* A body that leaks the tensor it builds, when traced, to a reference: a stale handle. *)
let leaked : Tensor.t option ref = ref None

let%op leaky () x =
  let y = x + 1.0 in
  [%oc leaked := Some y];
  y

(* A body reading the committed dims of a tensor that is open when the closure is built: the trace
   cannot run it (inference cannot close in a trace), and the failed read refuses the rewind. *)
let%op reads_dims c () x =
  ignore (Shape.to_dims c.Tensor.shape : int array);
  x + c

let%op reads_dims_plain c () x =
  ignore (Shape.to_dims c.Tensor.shape : int array);
  x + c
[@@ocannl.no_scheme]

(* {3 Option-valued tensor parameters (gh-ocannl-1792)}

   One scheme per presence pattern: which annotated option-valued tensor parameters are [Some]. *)

(* The mask meets a parameter of width 4 only when present. *)
let%op masked () ?(mask : Tensor.t option) x =
  Int.incr runs;
  let y = { mw; o = [ 4 ] } * x in
  match mask with None -> y | Some m -> y *. (m + { mb = 0.; o = [ 4 ] })

let%op masked_plain () ?(mask : Tensor.t option) x =
  Int.incr runs;
  let y = { mw; o = [ 4 ] } * x in
  match mask with None -> y | Some m -> y *. (m + { mb = 0.; o = [ 4 ] })
[@@ocannl.no_scheme]

(* Two option-valued parameters, one labelled and one optional: four patterns. *)
let%op stepped () ~(train_step : Tensor.t option) ?(mask : Tensor.t option) x =
  let x = match train_step with None -> x | Some s -> x + s in
  match mask with None -> x | Some m -> x *. m

(* A contradiction only the masked pattern reaches, whatever the arguments. *)
let%op broken_when_masked () ?(mask : Tensor.t option) x =
  match mask with None -> x | Some m -> x + m + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })

let%op broken_when_masked_plain () ?(mask : Tensor.t option) x =
  match mask with None -> x | Some m -> x + m + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })
[@@ocannl.no_scheme]

(* A refusal confined to the training pattern (the body refuses to run on tracers), and a random
   draw confined to it, which is schemed (gh-ocannl-1890). *)
let%op picky () ~(train_step : Tensor.t option) x =
  match train_step with
  | None -> x
  | Some s ->
      [%oc if Shape.tracing_in_progress () then failwith "picky: no tracers"];
      x + s

let%op noisy_step () ~(train_step : Tensor.t option) x =
  match train_step with None -> x | Some s -> x + uniform_at s

(* A labelled tensor that is only passed on: an annotation is evidence, a bare name is not. *)
let scale ~gamma x = TDSL.O.(gamma *. x)
let%op forwarded () ~(gamma : Tensor.t) x = scale ~gamma x
let%op forwarded_bare () ~gamma x = scale ~gamma x

(* Without a [()] stage: every pattern is traced at the first application. *)
let%op masked_late ?(mask : Tensor.t option) x =
  match mask with None -> x + x | Some m -> (x + x) *. m

let () =
  Verdict.case "the closure built at () traces once; each application checks first" (fun () ->
      Tensor.unsafe_reinitialize ();
      runs := 0;
      let lay, at_unit = events (fun () -> layer ()) in
      p "creating the closure traces its scheme once" (List.count at_unit ~f:(traced "layer") = 1);
      let x1 = leaf ~batch:[ 2 ] "x1" [ 3 ] and x2 = leaf ~batch:[ 5 ] "x2" [ 3 ] in
      let (_ : Tensor.t), at_first = events (fun () -> lay x1) in
      let (_ : Tensor.t), at_second = events (fun () -> lay x2) in
      p "each application checks the scheme"
        (List.count (at_first @ at_second) ~f:(checked "layer") = 2);
      p_none "the applications do not trace again" (at_first @ at_second) ~f:(traced "layer");
      p "the body ran once at the trace and once per application" (!runs = 3);
      let shapes_of lay =
        let a = leaf ~batch:[ 2 ] "a" [ 3 ] in
        let b = leaf ~batch:[ 5 ] "b" [ 3 ] in
        let ra = lay a and rb = lay b in
        List.map [ ra; rb ] ~f:(fun t -> Array.to_list (Shape.to_dims t.Tensor.shape))
      in
      Tensor.unsafe_reinitialize ();
      let schemed = shapes_of (layer ()) in
      Tensor.unsafe_reinitialize ();
      let plain = shapes_of (layer_plain ()) in
      p "the applications close as the opted-out twin's do"
        (List.equal (List.equal Int.equal) schemed plain));
  Verdict.case "an argument-independent error raises at (), naming the function" (fun () ->
      Tensor.unsafe_reinitialize ();
      let msg = error_message (fun () -> broken ()) in
      p "the closure's creation raises, naming the scheme of [broken]"
        (has_substring ~substring:"tracing the shape scheme broken" msg);
      Tensor.unsafe_reinitialize ();
      let f = broken_plain () in
      let msg = error_message (fun () -> f (leaf "x" [ 3 ])) in
      p "control (opted out): the error comes at the application, not naming the function"
        (Option.is_some msg && not (has_substring ~substring:"broken" msg)));
  Verdict.case "an argument-dependent error raises before the body runs" (fun () ->
      let conflict ~plain =
        Tensor.unsafe_reinitialize ();
        runs := 0;
        let lay = if plain then layer_plain () else layer () in
        ignore (lay (leaf ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t);
        let before = !runs in
        let msg = error_message (fun () -> lay (leaf ~batch:[ 2 ] "y" [ 5 ])) in
        (msg, !runs - before)
      in
      let msg, ran = conflict ~plain:false in
      p "a feature size conflicting with the shared parameter raises, naming the function"
        (has_substring ~substring:"applying the shape scheme layer to y" msg);
      p "the body did not run" (ran = 0);
      let msg, ran = conflict ~plain:true in
      p "control (opted out): the body's own error, after the body started"
        (Option.is_some msg && (not (has_substring ~substring:"shape scheme" msg)) && ran = 1);
      Tensor.For_testing.set_op_shape_schemes (Some `Off);
      let msg, _ = conflict ~plain:false in
      Tensor.For_testing.set_op_shape_schemes check_mode;
      p "control (schemes off): the body's own error"
        (Option.is_some msg && not (has_substring ~substring:"shape scheme" msg)));
  Verdict.case "which functions get a scheme" (fun () ->
      Tensor.unsafe_reinitialize ();
      (* Beside each subject, a schemed [layer] in the same window witnesses that the log
         records. *)
      let silent name mk apply =
        let (), evs =
          events (fun () ->
              let lay = layer () in
              ignore (lay (leaf "w" [ 2 ]) : Tensor.t);
              let f = mk () in
              ignore (apply f : Tensor.t))
        in
        p_none (name ^ ": no scheme, the function applies as before") evs ~f:(about name)
      in
      silent "with_step" with_step (fun f -> f ~train_step:(Some ()) (leaf "x" [ 2 ]));
      silent "with_mask" with_mask (fun f -> f ~mask:(leaf "m" [ 2 ]) (leaf "x" [ 2 ]));
      silent "rebound" rebound (fun f -> f ~s:(leaf "s" [ 2 ]) (leaf "x" [ 2 ]));
      silent "layer_plain" layer_plain (fun f -> f (leaf "x" [ 2 ]));
      let (), evs =
        events (fun () ->
            let f = affine () in
            ignore (f ~gamma:(leaf "g" [ 2 ]) ~beta:(leaf "b" [ 2 ]) (leaf "x" [ 2 ]) : Tensor.t))
      in
      p "labelled tensor parameters proven by their operand use get a scheme"
        (List.exists evs ~f:(traced "affine") && List.exists evs ~f:(checked "affine"));
      let (), evs =
        events (fun () ->
            let f = stack () in
            ignore (f (leaf ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t))
      in
      let names =
        List.filter_map evs ~f:(function
          | Tensor.Scheme.Traced s -> Some (Tensor.Scheme.label s)
          | _ -> None)
      in
      p "a returned function after its sub-layers is traced once they exist"
        (List.equal String.equal names [ "layer"; "layer"; "stack" ]));
  Verdict.case "a refused trace leaves the function unschemed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let (), evs =
        events (fun () ->
            let f = sets_outside r () in
            ignore (f (leaf "x" [ 4 ]) : Tensor.t))
      in
      p "setting a capture reference made outside: unschemed, by the trace's refusal"
        (List.exists evs ~f:(function
          | Tensor.Scheme.Unschemed (n, why) ->
              String.equal n "sets_outside" && String.is_substring why ~substring:"made outside it"
          | _ -> false));
      p_none "and the application ran unchecked" evs ~f:(fun e ->
          traced "sets_outside" e || checked "sets_outside" e);
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let msg =
        error_message (fun () ->
            Tensor.Scheme.trace ~label:"explicit" 1 (function
              | [ x ] -> [ sets_outside r () x ]
              | _ -> assert false))
      in
      p "control: the explicit trace of the same function still raises a shape error"
        (has_substring ~substring:"made outside it" msg));
  Verdict.case "random number generation is schemed; an explicit instance refuses it" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f, at_unit = events (fun () -> noisy ()) in
      let (_ : Tensor.t), at_apply = events (fun () -> f (leaf "x" [ 4 ])) in
      p "the closure's trace succeeds" (List.exists at_unit ~f:(traced "noisy"));
      p_none "and nothing is left unschemed" at_unit ~f:(function
        | Tensor.Scheme.Unschemed _ -> true
        | _ -> false);
      p "the application checks the scheme" (List.exists at_apply ~f:(checked "noisy"));
      let instance ?sabotage () =
        Tensor.unsafe_reinitialize ();
        let body = function [ x ] -> [ drawn_plain x ] | _ -> assert false in
        let s = Tensor.Scheme.trace ~label:"explicit" 1 body in
        let x = leaf "x" [ 12 ] in
        let r = Shape.make ~debug_name:"r" ~id:(-1) () in
        let inst () =
          match sabotage with
          | None -> Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:[ r ]
          | Some sabotage ->
              Tensor.For_testing.instantiate_scheme ~sabotage s ~args:[ x.shape ] ~results:[ r ]
        in
        match outcome inst with
        | Error msg -> `Refused msg
        | Ok () -> (
            match outcome (fun () -> ignore (Shape.to_dims r : int array)) with
            | Ok () -> `Closed
            | Error msg -> `Closure_failed msg)
      in
      p "the explicit trace succeeds" (Result.is_ok (outcome (fun () -> instance ())));
      p "its explicit instance is refused, naming the scheme and the remedy"
        (match instance () with
        | `Refused msg ->
            Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
            String.is_substring msg ~substring:"applying the shape scheme explicit"
            && String.is_substring msg ~substring:"draws random numbers"
            && String.is_substring msg ~substring:"apply the function inline"
        | `Closed | `Closure_failed _ -> false);
      p "control (no refusal): the closure fails at the draw's hole, settling no precision"
        (match instance ~sabotage:No_draw_refusal () with
        | `Closure_failed msg ->
            Stdio.eprintf "(not part of the golden) closure refused: %s\n%!" msg;
            String.is_substring msg ~substring:"read the coefficient of one of its random draws"
        | `Refused _ | `Closed -> false));
  Verdict.case "a check leaves the session as it was" (fun () ->
      Tensor.unsafe_reinitialize ();
      let lay = layer () in
      ignore (lay (leaf ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t);
      let h =
        Tensor.Scheme.For_ppx.closure ~name:"probe" ~arity:1 (function
          | [ x ] -> ignore (lay x : Tensor.t)
          | _ -> assert false)
      in
      let ok = leaf ~batch:[ 5 ] "ok" [ 3 ] and bad = leaf ~batch:[ 5 ] "bad" [ 6 ] in
      let ck = Tensor.checkpoint ~label:"before the checks" () in
      let (), evs = events (fun () -> Tensor.Scheme.For_ppx.apply h [ ok ] (fun () -> ())) in
      p "the check of a fitting argument ran" (List.exists evs ~f:(checked "probe"));
      p_empty "and left the session as it was" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p "the check of a conflicting argument raises, naming the scheme"
        (has_substring ~substring:"applying the shape scheme probe to bad"
           (error_message (fun () -> Tensor.Scheme.For_ppx.apply h [ bad ] (fun () -> ()))));
      p_empty "and left the session as it was too" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck));
  Verdict.case "a trace's discards are held weakly" (fun () ->
      Tensor.unsafe_reinitialize ();
      Gc_witness.settle ();
      let sizes0 = Tensor.For_testing.discarded_registry_sizes () in
      for _ = 1 to 20 do
        ignore (layer () : Tensor.t -> Tensor.t)
      done;
      Gc_witness.settle ();
      let sizes1 = Tensor.For_testing.discarded_registry_sizes () in
      p_all "twenty traces leave no record in any discard registry once unreachable"
        (List.zip_exn sizes0 sizes1) ~f:(fun ((_, n0), (_, n1)) -> n1 = n0);
      let f = leaky () in
      let stale = Option.value_exn !leaked in
      ignore (f (leaf "x" [ 2 ]) : Tensor.t);
      p "a handle leaked from a trace is still refused, naming the trace"
        (has_substring
           ~substring:"discarded by the rewind after the trace of the shape scheme leaky"
           (error_message (fun () -> TDSL.O.(stale + stale)))));
  Verdict.case "a refused rewind does not degrade silently" (fun () ->
      Tensor.unsafe_reinitialize ();
      let msg = error_message (fun () -> reads_dims (leaf "c" [ 3 ]) ()) in
      p "building the closure raises, naming the function and the opt-out"
        (has_substring ~substring:"the trace of the shape scheme reads_dims was refused" msg
        && has_substring ~substring:"[@@ocannl.no_scheme]" msg);
      Tensor.unsafe_reinitialize ();
      p "control (opted out): the same function builds and applies"
        (Result.is_ok (outcome (fun () -> reads_dims_plain (leaf "c" [ 3 ]) () (leaf "x" [ 3 ])))));
  Verdict.case "a function without a () stage traces at its first application" (fun () ->
      Tensor.unsafe_reinitialize ();
      (* A schemed [layer] built in the same window witnesses that the log records. *)
      let double, at_definition =
        events (fun () ->
            ignore (layer () : Tensor.t -> Tensor.t);
            let%op double x = x + x in
            double)
      in
      p_none "nothing is traced at the definition" at_definition ~f:(about "double");
      let (_ : Tensor.t), first = events (fun () -> double (leaf "x" [ 2 ])) in
      let (_ : Tensor.t), second = events (fun () -> double (leaf "y" [ 3 ])) in
      p "the first application traces, every application checks"
        (List.count first ~f:(traced "double") = 1
        && List.count (first @ second) ~f:(checked "double") = 2);
      p_none "the second application does not trace again" second ~f:(traced "double");
      Tensor.unsafe_reinitialize ();
      let (_ : Tensor.t), after_reset = events (fun () -> double (leaf "z" [ 4 ])) in
      p "after a session reset, the scheme is traced again"
        (List.count after_reset ~f:(traced "double") = 1));
  Verdict.case "applications across a closure" (fun () ->
      Tensor.unsafe_reinitialize ();
      let lay = layer () in
      let r = lay (leaf ~batch:[ 2 ] "x" [ 3 ]) in
      ignore (Shape.to_dims r.Tensor.shape : int array);
      p "after the closure, an application with the shared feature size succeeds"
        (Result.is_ok (outcome (fun () -> lay (leaf ~batch:[ 7 ] "x2" [ 3 ]))));
      p "after the closure, a conflicting feature size is refused, naming the function"
        (has_substring ~substring:"applying the shape scheme layer"
           (error_message (fun () -> lay (leaf ~batch:[ 7 ] "y" [ 5 ])))));
  Verdict.case "a closure built inside a body" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f, at_unit = events (fun () -> outer ()) in
      p "inside the outer trace, the inner closure stays unschemed"
        (List.exists at_unit ~f:(function
          | Tensor.Scheme.Unschemed (n, why) ->
              String.equal n "const_layer"
              && String.is_substring why ~substring:"inside a shape-scheme trace"
          | _ -> false));
      let (_ : Tensor.t), at_apply = events (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])) in
      p "at the application, the inner closure is traced after the outer check, which entails its"
        (match List.filter at_apply ~f:(fun e -> about "const_layer" e || about "outer" e) with
        | [ a; b ] -> checked "outer" a && traced "const_layer" b
        | _ -> false);
      Tensor.unsafe_reinitialize ();
      let f = builds_broken () in
      p "a contradiction in a closure built and never applied still raises, naming the closure"
        (has_substring ~substring:"tracing the shape scheme broken"
           (error_message (fun () -> f (leaf "x" [ 3 ])))));
  Verdict.case "a checked application vouches for the applications in its body" (fun () ->
      let count name evs = List.count evs ~f:(checked name) in
      let x () = leaf ~batch:[ 2 ] "x" [ 4 ] in
      Tensor.unsafe_reinitialize ();
      let s = stack () in
      let (_ : Tensor.t), evs = events (fun () -> s (x ())) in
      p "the outer application is checked, once" (count "stack" evs = 1);
      p "and the layers in its body are not" (count "layer" evs = 0);
      let refused_message app =
        Tensor.unsafe_reinitialize ();
        let f = app () in
        let y = f (x ()) in
        ignore (Shape.to_dims y.Tensor.shape : int array);
        error_message (fun () -> f (leaf ~batch:[ 7 ] "y" [ 5 ]))
      in
      p "a nested argument error is still refused, naming the outer function"
        (has_substring ~substring:"applying the shape scheme stack to y" (refused_message stack));
      Tensor.For_testing.with_nested_rechecks (fun () ->
          Tensor.unsafe_reinitialize ();
          let s = stack () in
          let (_ : Tensor.t), evs = events (fun () -> s (x ())) in
          p "the control checks the layers too" (count "stack" evs = 1 && count "layer" evs = 2);
          p "and its nested error names the same function: the outer check came first"
            (has_substring ~substring:"applying the shape scheme stack to y" (refused_message stack)));
      Tensor.unsafe_reinitialize ();
      let plain = stack_plain () in
      let (_ : Tensor.t), evs = events (fun () -> plain (x ())) in
      p "under an opted-out outer function, the layers are checked" (count "layer" evs = 2);
      p "and an argument error names the layer"
        (has_substring ~substring:"applying the shape scheme layer to y"
           (refused_message stack_plain));
      Tensor.unsafe_reinitialize ();
      let through = through_plain () in
      let (_ : Tensor.t), evs = events (fun () -> through (x ())) in
      p "an opted-out function under a checked one: its layers are entailed all the same"
        (count "through_plain" evs = 1 && count "layer" evs = 0);
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "r" in
      let refused, _ = events (fun () -> refused_outer r ()) in
      let (_ : Tensor.t), evs = events (fun () -> refused (x ())) in
      p "under a refused outer function, the layer is checked"
        (count "refused_outer" evs = 0 && count "layer" evs = 1);
      Tensor.unsafe_reinitialize ();
      let f = stack () in
      let (_ : Tensor.t), evs = events (fun () -> f (x ())) in
      let lay = layer () in
      let (_ : Tensor.t), after = events (fun () -> lay (x ())) in
      p "the skip ends with the body: a later application is checked"
        (count "stack" evs = 1 && count "layer" after = 1);
      Tensor.unsafe_reinitialize ();
      let f = raising_outer () in
      let lay = layer () in
      raise_in_body := true;
      let raised = Result.is_error (Or_error.try_with (fun () -> f (x ()))) in
      raise_in_body := false;
      let (_ : Tensor.t), after = events (fun () -> lay (x ())) in
      p "and with a body that raises" (raised && count "layer" after = 1);
      Tensor.unsafe_reinitialize ();
      let f = over_late () in
      let (_ : Tensor.t), evs = events (fun () -> f (x ())) in
      p "a function without a [()] stage, applied in a checked body: the outer one is checked"
        (count "over_late" evs = 1);
      p_none "and the entailed one is neither traced nor checked" evs ~f:(about "nested_late");
      let (_ : Tensor.t), evs = events (fun () -> nested_late (x ())) in
      p "and its own application traces and checks it"
        (List.exists evs ~f:(traced "nested_late") && count "nested_late" evs = 1);
      Tensor.unsafe_reinitialize ();
      let f = over_twice_when_masked () in
      p "a first application in a checked body still validates the patterns it does not take"
        (has_substring ~substring:"tracing the shape scheme twice_when_masked [mask]"
           (error_message (fun () -> f (x ())))));
  Verdict.case "schemes change no id and no value" (fun () ->
      let run ~on =
        Tensor.For_testing.set_op_shape_schemes (Some (if on then `Check else `Off));
        Tensor.unsafe_reinitialize ();
        let first = Tensor.term ~grad_spec:Prohibit_grad ~label:[ "first" ] () in
        let first_var = Row.var_mark () in
        let lay1 = layer () in
        let lay2 = affine () in
        let x = Tensor.term_init [| 0.5; -1.0; 2.0 |] ~grad_spec:Prohibit_grad ~label:[ "x" ] () in
        let g = TDSL.param "g" ~output_dims:[ 4 ] () in
        let b = TDSL.param "b" ~output_dims:[ 4 ] () in
        let%op loss = lay2 ~gamma:g ~beta:b (lay1 x) ++ "...|... => 0" in
        Set.iter loss.Tensor.params ~f:(fun t ->
            Option.iter t.Tensor.diff ~f:(fun d -> Ocannl.Train.set_materialized d.Tensor.grad));
        let ctx = Ocannl.Train.update_once (Context.auto ()) loss in
        let params = Set.to_list loss.Tensor.params in
        let values t = Array.to_list (Context.get_values ctx t.Tensor.value) in
        let grads t =
          Option.value_map t.Tensor.diff ~default:[] ~f:(fun d ->
              Array.to_list (Context.get_values ctx d.Tensor.grad))
        in
        Tensor.For_testing.set_op_shape_schemes check_mode;
        ( ( Tensor.get_next_id (),
            loss.Tensor.value.Ir.Tnode.uid - first.Tensor.value.Ir.Tnode.uid,
            Row.var_mark () - first_var ),
          List.map params ~f:(fun t -> t.Tensor.value.Ir.Tnode.id),
          values loss :: List.concat_map params ~f:(fun t -> [ values t; grads t ]) )
      in
      let id1 (a, _, _) = a and uid_span (_, b, _) = b and var_span (_, _, c) = c in
      let next_on, ids_on, vals_on = run ~on:true in
      let next_off, ids_off, vals_off = run ~on:false in
      Stdio.eprintf "(not part of the golden) next ids: on %d, off %d; values: %s\n%!" (id1 next_on)
        (id1 next_off)
        (String.concat ~sep:" | "
           (List.map vals_on ~f:(fun l -> String.concat ~sep:" " (List.map l ~f:Float.to_string))));
      p "the session hands out the same ids with schemes on and off"
        (id1 next_on = id1 next_off && List.equal Int.equal ids_on ids_off);
      (* Node uids are not reserved (gh-ocannl-2053): the traces' discarded nodes take uids from the
         session counter, and the claim below compares two programs whose uids differ. *)
      p "control: the traces shift the node uids, relative to the run's first node"
        (uid_span next_on <> uid_span next_off);
      p "and the same shape variable ids, relative to the run's start"
        (var_span next_on = var_span next_off);
      p "the loss, the parameters (random initialization included) and their gradients are the same"
        (List.equal (List.equal Float.equal) vals_on vals_off);
      let next_sabotaged, _, _ =
        Tensor.For_testing.without_id_neutrality (fun () -> run ~on:true)
      in
      p "control (no id neutrality): the traces shift the session's ids"
        (id1 next_sabotaged <> id1 next_off);
      p "control (no id neutrality): and its shape variable ids"
        (var_span next_sabotaged <> var_span next_off))

let () =
  let mask_ = "masked [mask]" and no_mask = "masked [no mask]" in
  Verdict.case "option-valued parameters: one scheme per presence pattern, traced at ()" (fun () ->
      Tensor.unsafe_reinitialize ();
      let _, evs = events (fun () -> masked ()) in
      p "both patterns are traced at ()"
        (List.count evs ~f:(traced mask_) = 1 && List.count evs ~f:(traced no_mask) = 1);
      let _, evs = events (fun () -> stepped ()) in
      p_all "two option-valued parameters: all four patterns are traced at ()"
        [
          "stepped [train_step, mask]";
          "stepped [train_step, no mask]";
          "stepped [no train_step, mask]";
          "stepped [no train_step, no mask]";
        ] ~f:(fun l -> List.count evs ~f:(traced l) = 1);
      (* A schemed [layer] in the same window witnesses that the log records. *)
      let _, evs =
        events (fun () ->
            ignore (layer () : Tensor.t -> Tensor.t);
            masked_plain ())
      in
      p_none "control (opted out): nothing is traced" evs ~f:(fun e ->
          about mask_ e || about no_mask e || about "masked_plain" e));
  Verdict.case "option-valued parameters: each application checks its own pattern" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f = masked () in
      let x () = leaf ~batch:[ 2 ] "x" [ 3 ] in
      let (_ : Tensor.t), evs1 = events (fun () -> f ~mask:(leaf "m1" [ 4 ]) (x ())) in
      let (_ : Tensor.t), evs2 = events (fun () -> f ~mask:(leaf "m2" [ 4 ]) (x ())) in
      let (_ : Tensor.t), evs3 = events (fun () -> f (x ())) in
      p "two masked applications, different masks: the masked scheme is checked twice"
        (List.count (evs1 @ evs2) ~f:(checked mask_) = 2);
      p_none "and neither checks nor traces the unmasked one" (evs1 @ evs2) ~f:(fun e ->
          checked no_mask e || traced no_mask e || traced mask_ e);
      p "an unmasked application checks the unmasked scheme"
        (List.count evs3 ~f:(checked no_mask) = 1);
      p_none "and not the masked one" evs3 ~f:(checked mask_);
      let g = stepped () in
      let (_ : Tensor.t), evs =
        events (fun () -> g ~train_step:(Some (leaf "s" [ 3 ])) (leaf "y" [ 3 ]))
      in
      p "a labelled option and an omitted optional select their pattern"
        (List.count evs ~f:(checked "stepped [train_step, no mask]") = 1));
  Verdict.case "option-valued parameters: an argument-dependent error of one pattern" (fun () ->
      let conflict ~plain =
        Tensor.unsafe_reinitialize ();
        runs := 0;
        let f = if plain then masked_plain () else masked () in
        let unmasked = outcome (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])) in
        let before = !runs in
        let msg =
          error_message (fun () -> f ~mask:(leaf "wide" [ 5 ]) (leaf ~batch:[ 2 ] "y" [ 3 ]))
        in
        (Result.is_ok unmasked, msg, !runs - before)
      in
      let unmasked_ok, msg, ran = conflict ~plain:false in
      p "the unmasked application succeeds" unmasked_ok;
      p "a mask conflicting with the masked pattern raises, naming the pattern"
        (has_substring ~substring:"applying the shape scheme masked [mask] to wide" msg);
      p "the body did not run" (ran = 0);
      let _, msg, ran = conflict ~plain:true in
      p "control (opted out): the body's own error, after the body started"
        (Option.is_some msg && (not (has_substring ~substring:"shape scheme" msg)) && ran = 1));
  Verdict.case "option-valued parameters: an argument-independent error of one pattern" (fun () ->
      Tensor.unsafe_reinitialize ();
      let msg = error_message (fun () -> broken_when_masked ()) in
      p "the closure's creation raises, naming the masked pattern"
        (has_substring ~substring:"tracing the shape scheme broken_when_masked [mask]" msg);
      Tensor.unsafe_reinitialize ();
      let msg =
        error_message (fun () ->
            ignore
              (Tensor.Scheme.check_definition ~name:"broken_when_masked" (fun () ->
                   ignore (broken_when_masked () : ?mask:Tensor.t -> Tensor.t -> Tensor.t))
                : Tensor.Scheme.t list))
      in
      p "check_definition reports it, naming the definition and the pattern"
        (has_substring
           ~substring:
             "checking the definition broken_when_masked: tracing the shape scheme \
              broken_when_masked [mask]"
           msg);
      Tensor.unsafe_reinitialize ();
      p "control (opted out): the closure builds, and an unmasked application succeeds"
        (Result.is_ok
           (outcome (fun () -> broken_when_masked_plain () (leaf ~batch:[ 2 ] "x" [ 3 ])))));
  Verdict.case "option-valued parameters: a refused pattern degrades alone" (fun () ->
      let refused name f apply_eval =
        Tensor.unsafe_reinitialize ();
        let g, at_unit = events f in
        let (_ : Tensor.t), at_apply = events (fun () -> apply_eval g) in
        p
          (name ^ ": the training pattern is unschemed")
          (List.exists at_unit ~f:(function
            | Tensor.Scheme.Unschemed (n, _) -> String.equal n (name ^ " [train_step]")
            | _ -> false));
        p
          (name ^ ": the evaluation pattern is traced at () and checked at its application")
          (List.exists at_unit ~f:(traced (name ^ " [no train_step]"))
          && List.exists at_apply ~f:(checked (name ^ " [no train_step]")))
      in
      refused "picky" picky (fun g -> g ~train_step:None (leaf "x" [ 3 ]));
      Tensor.unsafe_reinitialize ();
      let (_ : _), at_unit = events noisy_step in
      p "noisy_step: both patterns, the random draw's included, are traced at () (gh-ocannl-1890)"
        (List.exists at_unit ~f:(traced "noisy_step [train_step]")
        && List.exists at_unit ~f:(traced "noisy_step [no train_step]"));
      Tensor.unsafe_reinitialize ();
      let g = picky () in
      let lay = layer () in
      (* The schemed [layer]'s check witnesses that the log records. *)
      let (_ : Tensor.t), evs =
        events (fun () ->
            ignore (lay (leaf "w" [ 3 ]) : Tensor.t);
            g ~train_step:(Some (leaf "s" [ 3 ])) (leaf "x" [ 3 ]))
      in
      p_none "a training application runs unchecked" evs ~f:(fun e ->
          checked "picky [train_step]" e || traced "picky [train_step]" e));
  Verdict.case
    "option-valued parameters: tracing begins with every pattern, re-tracing is per pattern"
    (fun () ->
      Tensor.unsafe_reinitialize ();
      let late_mask = "masked_late [mask]" and late_no_mask = "masked_late [no mask]" in
      let (_ : Tensor.t), first =
        events (fun () -> masked_late ~mask:(leaf "m" [ 2 ]) (leaf "x" [ 2 ]))
      in
      let (_ : Tensor.t), second = events (fun () -> masked_late (leaf "y" [ 2 ])) in
      p "without a () stage, the first application traces every pattern"
        (List.count first ~f:(traced late_mask) = 1 && List.count first ~f:(traced late_no_mask) = 1);
      p "the other pattern's first application checks"
        (List.count second ~f:(checked late_no_mask) = 1);
      p_none "and does not trace" second ~f:(traced late_no_mask);
      Tensor.unsafe_reinitialize ();
      let (_ : Tensor.t), evs =
        events (fun () -> masked_late ~mask:(leaf "m2" [ 3 ]) (leaf "x2" [ 3 ]))
      in
      p "after a session reset, the applied pattern alone is traced again, and checked"
        (List.count evs ~f:(traced late_mask) = 1 && List.count evs ~f:(checked late_mask) = 1);
      p_none "the other pattern is not traced" evs ~f:(traced late_no_mask));
  Verdict.case "option-valued parameters: a check and the traces leave the session as it was"
    (fun () ->
      Tensor.unsafe_reinitialize ();
      let ck = Tensor.checkpoint ~label:"before the variant traces" () in
      let next_id = Tensor.get_next_id () in
      ignore
        (Tensor.Scheme.check_definition ~name:"stepped" (fun () ->
             ignore
               (stepped () : train_step:Tensor.t option -> ?mask:Tensor.t -> Tensor.t -> Tensor.t))
          : Tensor.Scheme.t list);
      (* The failing one inside a definition check too: its closure's [()] parameters are session
         tensors. *)
      (try
         ignore
           (Tensor.Scheme.check_definition ~name:"broken_when_masked" (fun () ->
                ignore (broken_when_masked () : ?mask:Tensor.t -> Tensor.t -> Tensor.t))
             : Tensor.Scheme.t list)
       with Row.Shape_error _ -> ());
      p "the traces allocated no session id" (Tensor.get_next_id () = next_id);
      p_empty "successful and failed variant traces left the session as it was"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck));
  Verdict.case "option-valued parameters: which get a scheme" (fun () ->
      Tensor.unsafe_reinitialize ();
      let (), evs =
        events (fun () ->
            let f = forwarded () in
            ignore (f ~gamma:(leaf "g" [ 2 ]) (leaf "x" [ 2 ]) : Tensor.t))
      in
      p "a labelled tensor annotated [Tensor.t] gets a scheme without operand evidence"
        (List.exists evs ~f:(traced "forwarded") && List.exists evs ~f:(checked "forwarded"));
      let (), evs =
        events (fun () ->
            let lay = layer () in
            ignore (lay (leaf "w" [ 2 ]) : Tensor.t);
            let f = forwarded_bare () in
            ignore (f ~gamma:(leaf "g" [ 2 ]) (leaf "x" [ 2 ]) : Tensor.t))
      in
      p "control: the same parameter unannotated stays unschemed, beside a schemed witness"
        (List.exists evs ~f:(traced "layer") && not (List.exists evs ~f:(about "forwarded_bare"))))

(* {2 External capture references (gh-ocannl-1791)} *)

let () =
  Verdict.case "a capture reference made outside the function: schemed, bound per application"
    (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let f, at_unit = events (fun () -> summed r ()) in
      p "the closure's trace succeeds" (List.exists at_unit ~f:(traced "summed"));
      p_none "and nothing is left unschemed" at_unit ~f:(function
        | Tensor.Scheme.Unschemed _ -> true
        | _ -> false);
      let xv = Array.init 10 ~f:(fun i -> Float.of_int (i + 1) *. 10.) in
      let x = TDSL.ndarray xv ~label:[ "x" ] ~output_dims:[ 2; 5 ] () in
      let y, at_first = events (fun () -> f x) in
      p "the application checks the scheme" (List.exists at_first ~f:(checked "summed"));
      let ctx = Ocannl.Train.forward_once (Context.auto ()) y in
      let yv = Context.get_values ctx y.value in
      let expected = [| 150.; 400. |] in
      p_close "executed: each output is the sum of its row, against the closed form" yv expected
        ~tol:(fun _ -> 1e-3);
      p "the closure published the reference's solution, the size of the summed axis"
        (Option.equal Int.equal r.var_ref.solved_dim (Some 5));
      (* The reference changed since the trace (it was unbound then): a conflicting application is
         refused by the check, before the body runs. *)
      let runs_before = !summed_runs in
      let msg = error_message (fun () -> f (leaf "bad" [ 3; 7 ])) in
      p "a conflicting application is refused at the application, naming the scheme"
        (has_substring ~substring:"applying the shape scheme summed to bad" msg);
      p "before the body ran" (!summed_runs = runs_before);
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let g = summed_plain r () in
      ignore (Shape.to_dims (g (leaf "x" [ 2; 5 ])).Tensor.shape : int array);
      let runs_before = !summed_runs in
      let msg = error_message (fun () -> g (leaf "bad" [ 3; 7 ])) in
      p "control (no scheme): the body refuses it, unnamed"
        (Option.is_some msg && not (has_substring ~substring:"applying the shape scheme" msg));
      p "control (no scheme): after the body ran" (!summed_runs = runs_before + 1));
  Verdict.case "a capture input's checks leave the session as it was" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let h =
        Tensor.Scheme.For_ppx.closure ~name:"probe_capture" ~arity:1 (function
          | [ x ] ->
              ignore (NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" x () : Tensor.t)
          | _ -> assert false)
      in
      (* An application in open work binds the reference. *)
      ignore
        (NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" (leaf "x0" [ 2; 5 ]) ()
          : Tensor.t);
      let ok = leaf "ok" [ 3; 5 ] and bad = leaf "bad" [ 3; 7 ] in
      let ck = Tensor.checkpoint ~label:"before the checks" () in
      let (), evs = events (fun () -> Tensor.Scheme.For_ppx.apply h [ ok ] (fun () -> ())) in
      p "the check of a fitting argument ran" (List.exists evs ~f:(checked "probe_capture"));
      p_empty "and left the session as it was" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p
        "the check of an argument conflicting with the reference's binding raises, naming the \
         scheme"
        (has_substring ~substring:"applying the shape scheme probe_capture to bad"
           (error_message (fun () -> Tensor.Scheme.For_ppx.apply h [ bad ] (fun () -> ()))));
      p_empty "and left the session as it was too" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck));
  Verdict.case "a capture input changes no id" (fun () ->
      let run ~schemed =
        Tensor.unsafe_reinitialize ();
        let first_var = Row.var_mark () in
        let r = Shape.get_variable_ref "b" in
        let f = if schemed then summed r () else summed_plain r () in
        let (_ : Tensor.t) = f (leaf "x1" [ 2; 5 ]) in
        let y2 = f (leaf "x2" [ 3; 5 ]) in
        ignore (Shape.to_dims y2.Tensor.shape : int array);
        (Tensor.get_next_id (), Row.var_mark () - first_var)
      in
      let next_on, vars_on = run ~schemed:true in
      let next_off, vars_off = run ~schemed:false in
      p "the schemed function hands out the session ids its unschemed twin does" (next_on = next_off);
      p "and the same shape variable ids" (vars_on = vars_off);
      let next_sabotaged, _ =
        Tensor.For_testing.without_id_neutrality (fun () -> run ~schemed:true)
      in
      p "control (no id neutrality): the trace shifts the session's ids" (next_sabotaged <> next_off))

(* A refused trace says so on stderr (gh-ocannl-1791), once per function and reason, naming the
   function; an opted-out or switched-off function, and a closure built inside a trace, do not. *)
let () =
  Verdict.case "a refused trace warns, once" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let evs = ref [] in
      let (), lines =
        Stderr_capture.lines (fun () ->
            Tensor.For_testing.observe_schemes
              (fun () ->
                ignore ((sets_warned r ()) (leaf "x" [ 4 ]) : Tensor.t);
                ignore ((sets_warned r ()) (leaf "y" [ 4 ]) : Tensor.t);
                ignore ((sets_quiet r ()) (leaf "z" [ 4 ]) : Tensor.t);
                ignore ((outer_quiet ()) (leaf ~batch:[ 2 ] "w" [ 3 ]) : Tensor.t);
                Tensor.For_testing.set_op_shape_schemes (Some `Off);
                Exn.protect
                  ~f:(fun () -> ignore ((sets_off r ()) (leaf "v" [ 4 ]) : Tensor.t))
                  ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes check_mode))
              ~on_event:(fun e -> evs := e :: !evs))
      in
      let warnings = List.filter lines ~f:(String.is_prefix ~prefix:"OCANNL warning:") in
      let about name l = String.is_substring l ~substring:("the shape scheme of " ^ name ^ " ") in
      p "the refused function was left unschemed twice"
        (List.count !evs ~f:(function
           | Tensor.Scheme.Unschemed (n, _) -> String.equal n "sets_warned"
           | _ -> false)
        = 2);
      p "one warning names it, on stderr" (List.count warnings ~f:(about "sets_warned") = 1);
      p_exists "with the trace's reason" warnings ~f:(fun l ->
          about "sets_warned" l && String.is_substring l ~substring:"made outside it");
      p "no warning for the opted-out function" (List.count warnings ~f:(about "sets_quiet") = 0);
      p "nor for a closure built inside a trace"
        (List.exists !evs ~f:(function
           | Tensor.Scheme.Unschemed (n, _) -> String.equal n "const_inner"
           | _ -> false)
        && List.count warnings ~f:(about "const_inner") = 0);
      p "nor with the schemes off" (List.count warnings ~f:(about "sets_off") = 0))

(* {2 Random draws (gh-ocannl-1890)}

   A packed conversion's coefficient (values per 128-bit block) is a lazy over its result node's
   precision, which consumers set top-down as they register and closure forces: it reads the call
   site's precision. The scheme carries a hole in its place, never read; each application's body
   registers its own conversion node, which fills it. *)

let draw_counter_blocks (y : Tensor.t) (x : Tensor.t) =
  let u =
    List.find_map_exn y.Tensor.children ~f:(fun c ->
        if phys_equal c.Tensor.subtensor x then None else Some c.subtensor)
  in
  let counter = (List.hd_exn u.Tensor.children).subtensor in
  Array.fold (Shape.to_dims counter.Tensor.shape) ~init:1 ~f:( * )

let () =
  Verdict.case "a random draw reads its call site's precision, schemed as inline" (fun () ->
      let n = 12 in
      let run ~schemed () =
        Tensor.For_testing.set_op_shape_schemes (Some (if schemed then `Check else `Off));
        Exn.protect
          ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes check_mode)
          ~f:(fun () ->
            Tensor.unsafe_reinitialize ();
            let f, at_unit = events (fun () -> drawn ()) in
            let xv = Array.init n ~f:(fun i -> Float.of_int (i + 1)) in
            let x1 = TDSL.ndarray xv ~label:[ "x1" ] ~output_dims:[ n ] () in
            let x2 = TDSL.ndarray xv ~label:[ "x2" ] ~output_dims:[ n ] () in
            Ir.Tnode.update_prec x2.value Ir.Ops.half;
            let y1, at1 = events (fun () -> f x1) in
            let y2, at2 = events (fun () -> f x2) in
            let ctx = Ocannl.Train.forward_once (Context.auto ()) y1 in
            let ctx = Ocannl.Train.forward_once ctx y2 in
            let values t = Array.to_list (Context.get_values ctx t.Tensor.value) in
            let evs = at_unit @ at1 @ at2 in
            ( (draw_counter_blocks y1 x1, draw_counter_blocks y2 x2),
              (values y1, values y2),
              (List.count evs ~f:(traced "drawn"), List.count evs ~f:(checked "drawn")) ))
      in
      let blocks_off, vals_off, _ = run ~schemed:false () in
      let blocks_on, vals_on, (traces, checks) = run ~schemed:true () in
      let show (a, b) = Int.to_string a ^ " and " ^ Int.to_string b in
      Stdio.eprintf "(not part of the golden) blocks: inline %s, schemed %s\n%!" (show blocks_off)
        (show blocks_on);
      p "inline: 12 values take 3 blocks at single precision and 2 at half"
        (Poly.equal blocks_off (3, 2));
      p "schemed: the function is traced once and each application checked"
        (traces = 1 && checks = 2);
      p "schemed: each application's counter has inline construction's block count"
        (Poly.equal blocks_on blocks_off);
      p "and draws inline construction's values"
        (Poly.equal (fst vals_on) (fst vals_off) && Poly.equal (snd vals_on) (snd vals_off));
      let blocks_memo =
        match Shape.with_memoized_draw_coefficient (run ~schemed:true) with
        | blocks, _, _ -> `Blocks blocks
        | exception (Row.Shape_error (msg, _) | Tensor.Session_error (msg, _) | Utils.User_error msg)
          ->
            `Failed msg
      in
      p
        "control (one coefficient memoized from the trace): the half application loses its block \
         count"
        (match blocks_memo with
        | `Blocks b ->
            Stdio.eprintf "(not part of the golden) memoized blocks: %s\n%!" (show b);
            not (Poly.equal b blocks_off)
        | `Failed msg ->
            Stdio.eprintf "(not part of the golden) memoized run refused: %s\n%!" msg;
            false));
  Verdict.case "tracing and checking a draw settle no precision" (fun () ->
      Tensor.unsafe_reinitialize ();
      let c0 = leaf "c0" [ 4 ] in
      (* An operation's result: its precision is a lazy that the product's forces. *)
      let c = TDSL.O.(c0 + c0) in
      let f, at_unit = events (fun () -> drawn_with c ()) in
      let (_ : Tensor.t), at_apply = events (fun () -> f (leaf "x" [ 4 ])) in
      p "the function is traced and its application checked"
        (List.exists at_unit ~f:(traced "drawn_with")
        && List.exists at_apply ~f:(checked "drawn_with"));
      p "the precision of the tensor in the draw's precision chain is still unsettled"
        (not (Lazy.is_val c.Tensor.value.Ir.Tnode.storage_prec)))

(* The trace's own nodes die with its rewind: a held scheme keeps none of them. *)
let[@inline never] held_draw_scheme ?sabotage () =
  let w = ref None in
  let body = function
    | [ x ] ->
        let y = drawn_plain x in
        let u =
          List.find_map_exn y.Tensor.children ~f:(fun c ->
              if phys_equal c.Tensor.subtensor x then None else Some c.subtensor)
        in
        w := Some (Gc_witness.witness u.Tensor.value);
        [ y ]
    | _ -> assert false
  in
  let s =
    match sabotage with
    | None -> Tensor.Scheme.trace ~label:"held" 1 body
    | Some sabotage -> Tensor.For_testing.trace_scheme ~sabotage ~label:"held" 1 body
  in
  (s, Option.value_exn !w)

let () =
  Verdict.case "a held scheme with a draw keeps none of its trace's nodes" (fun () ->
      Tensor.unsafe_reinitialize ();
      let s, w = held_draw_scheme () in
      p "the trace's draw node is reclaimed while its scheme is held" (Gc_witness.reclaimed w);
      ignore (Sys.opaque_identity s : Tensor.Scheme.t);
      Tensor.unsafe_reinitialize ();
      let s, w = held_draw_scheme ~sabotage:No_draw_holes () in
      p "control (the trace's own coefficients kept): the node is kept" (Gc_witness.kept w);
      ignore (Sys.opaque_identity s : Tensor.Scheme.t))
