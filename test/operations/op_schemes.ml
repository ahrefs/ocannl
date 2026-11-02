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

let traced name = function Tensor.Scheme.Traced n -> String.equal n name | _ -> false
let checked name = function Tensor.Scheme.Checked n -> String.equal n name | _ -> false

let about name = function
  | Tensor.Scheme.Traced n | Checked n | Unschemed (n, _) -> String.equal n name

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

(* A closure built inside the body: inside the outer trace, it stays unschemed. Its parameter has a
   constant initialization: a random one would make the outer trace refused (random number
   generation inside it). *)
let%op const_layer () x = { cw = 0.5; o = [ 4 ] } * x

let%op outer () x =
  let inner = const_layer () in
  inner x

(* Random number generation: a trace a scheme cannot carry. *)
let%op noisy () x = x + uniform ()

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

(* Refusals confined to the training pattern: a deterministic one (the body refuses to run on
   tracers) and random number generation. *)
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
        List.filter_map evs ~f:(function Tensor.Scheme.Traced n -> Some n | _ -> None)
      in
      p "a returned function after its sub-layers is traced once they exist"
        (List.equal String.equal names [ "layer"; "layer"; "stack" ]));
  Verdict.case "a refused trace leaves the function unschemed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let (), evs =
        events (fun () ->
            let f = noisy () in
            ignore (f (leaf "x" [ 4 ]) : Tensor.t))
      in
      p "random number generation: unschemed, by the trace's refusal"
        (List.exists evs ~f:(function
          | Tensor.Scheme.Unschemed (n, why) ->
              String.equal n "noisy" && String.is_substring why ~substring:"precision-dependent"
          | _ -> false));
      p_none "and the application ran unchecked" evs ~f:(fun e ->
          traced "noisy" e || checked "noisy" e);
      Tensor.unsafe_reinitialize ();
      let f = noisy () in
      let msg =
        error_message (fun () ->
            Tensor.Scheme.trace ~label:"explicit" 1 (function
              | [ x ] -> [ f x ]
              | _ -> assert false))
      in
      p "control: the explicit trace of the same function still raises a shape error"
        (has_substring ~substring:"precision-dependent" msg));
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
  Verdict.case "a trace's discarded nodes are held weakly" (fun () ->
      Tensor.unsafe_reinitialize ();
      Stdlib.Gc.full_major ();
      let nodes0, weak0 = Tensor.For_testing.discarded_registry_sizes () in
      for _ = 1 to 20 do
        ignore (layer () : Tensor.t -> Tensor.t)
      done;
      Stdlib.Gc.full_major ();
      let nodes1, weak1 = Tensor.For_testing.discarded_registry_sizes () in
      p "twenty traces leave no node in the per-node registry" (nodes1 = nodes0);
      p "and the weak registry keeps none of their nodes once unreachable" (weak1 = weak0);
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
      p "at the application, the inner closure is traced and checked, after the outer check"
        (match List.filter at_apply ~f:(fun e -> about "const_layer" e || about "outer" e) with
        | [ a; b; c ] -> checked "outer" a && traced "const_layer" b && checked "const_layer" c
        | _ -> false));
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
      p "and the same node uids, relative to the run's first node"
        (uid_span next_on = uid_span next_off);
      p "and the same shape variable ids, relative to the run's start"
        (var_span next_on = var_span next_off);
      p "the loss, the parameters (random initialization included) and their gradients are the same"
        (List.equal (List.equal Float.equal) vals_on vals_off);
      let next_sabotaged, _, _ =
        Tensor.For_testing.without_id_neutrality (fun () -> run ~on:true)
      in
      p "control (no id neutrality): the traces shift the session's ids"
        (id1 next_sabotaged <> id1 next_off);
      p "control (no id neutrality): and its node uids"
        (uid_span next_sabotaged <> uid_span next_off);
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
                : string list))
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
      refused "noisy_step" noisy_step (fun g -> g ~train_step:None (leaf "x" [ 3 ]));
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
          : string list);
      (* The failing one inside a definition check too: its closure's [()] parameters are session
         tensors. *)
      (try
         ignore
           (Tensor.Scheme.check_definition ~name:"broken_when_masked" (fun () ->
                ignore (broken_when_masked () : ?mask:Tensor.t -> Tensor.t -> Tensor.t))
             : string list)
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
