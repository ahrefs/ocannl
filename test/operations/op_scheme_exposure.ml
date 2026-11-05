(* gh-ocannl-1801: the scheme an [%op] closure traced is reachable. [Tensor.Scheme.traced_by]
   returns the schemes the closures built while it ran traced -- the ones their applications are
   checked against -- and an automatic scheme carries the function's results, read off what the
   traced body returns. The claims relate the exposed scheme to an explicit trace of the same
   closure, and its results to what the body builds: instantiating the scheme at an argument
   predicts the shape the body's result gets at that argument. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Nn = Ocannl.Nn_blocks

let%op layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })

(* Two results, the first one consumed by the second. *)
let%op pair ~hid () x =
  let h = { w; o = [ hid ] } * x in
  (h, relu h)

(* Its argument, returned as it is. *)
let%op passthrough () x = x

(* A value that is not a tensor or a tuple of tensors: an OCaml list ([%op] reads a list literal as
   a concatenation). *)
let%op listed () x = [%oc [ x; x ]]

(* No [()] stage: traced at the first application. *)
let%op doubled x = x + x

let input ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

(* A tensor with no shape information of its own, for a result to instantiate a scheme at. *)
let fresh name = Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ()
let shape (t : Tensor.t) = Shape.to_string_hum t.shape
let labels = List.map ~f:Tensor.Scheme.label

(* How many of the events while [f] runs were checks, and how many traces. *)
let checks_and_traces f =
  let checks = ref 0 and traces = ref 0 in
  Tensor.For_testing.observe_schemes f ~on_event:(function
    | Tensor.Scheme.Checked _ -> Int.incr checks
    | Traced _ -> Int.incr traces
    | Unschemed _ -> ());
  (!checks, !traces)

(* The schemes [build] traced, printed for the golden. *)
let traced build =
  let f, schemes = Tensor.Scheme.traced_by build in
  List.iter schemes ~f:(fun s -> Stdio.print_endline (Tensor.Scheme.to_string s));
  (f, schemes)

let () =
  Verdict.case "a closure's scheme, exposed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let l, schemes = traced (fun () -> layer ~hid:4 ()) in
      p "the closure built at () traced one scheme, under its function's name"
        (List.equal String.equal (labels schemes) [ "layer" ]);
      let s = List.hd_exn schemes in
      p "the scheme has the function's argument and its result"
        (Tensor.Scheme.arity s = 1 && Tensor.Scheme.num_results s = 1);
      (* The explicit trace of the closure is the same trace, run again. *)
      let explicit =
        Tensor.Scheme.trace ~label:"layer" 1 (function [ x ] -> [ l x ] | _ -> assert false)
      in
      p "it prints as the explicit trace of the closure does"
        (String.equal (Tensor.Scheme.to_string s) (Tensor.Scheme.to_string explicit));
      let x = input ~batch:[ 2 ] "x" [ 3 ] in
      let y = l x in
      let x' = input ~batch:[ 5; 7 ] "x'" [ 3 ] and r = fresh "r" in
      Tensor.Scheme.instantiate s ~args:[ x'.shape ] ~results:[ r.shape ];
      let y' = l x' in
      Stdio.printf "y: %s; at x' the scheme predicts %s, the body builds %s\n" (shape y) (shape r)
        (shape y');
      p "instantiated at an argument, the scheme predicts the shape the body's result gets"
        (String.equal (shape r) (shape y'));
      let checks, traces =
        checks_and_traces (fun () -> ignore (l (input ~batch:[ 3 ] "x''" [ 3 ]) : Tensor.t))
      in
      p "a later application is checked against the scheme exposed, tracing none"
        (checks = 1 && traces = 0));
  Verdict.case "results: a tuple, an argument, other values" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f, schemes = traced (fun () -> pair ~hid:4 ()) in
      let s = List.hd_exn schemes in
      p "a returned pair gives two results, in order, an intermediate among them"
        (Tensor.Scheme.num_results s = 2);
      let x = input ~batch:[ 2 ] "x" [ 3 ] and r1 = fresh "r1" and r2 = fresh "r2" in
      Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:[ r1.shape; r2.shape ];
      let h, out = f x in
      p "each predicts its component's shape"
        (String.equal (shape r1) (shape h) && String.equal (shape r2) (shape out));
      let g, schemes = traced (fun () -> passthrough ()) in
      let s = List.hd_exn schemes in
      p "a returned argument is the result" (Tensor.Scheme.num_results s = 1);
      let x = input ~batch:[ 2 ] "x2" [ 3 ] and r = fresh "r3" in
      Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:[ r.shape ];
      p "the result is the argument's shape" (String.equal (shape r) (shape x) && phys_equal (g x) x);
      let _, schemes = traced (fun () -> listed ()) in
      p "a list gives no results"
        (match schemes with [ s ] -> Tensor.Scheme.num_results s = 0 | _ -> false));
  Verdict.case "a copy of a made tensor's record is its result" (fun () ->
      (* Batch norm in training returns [{ result with forward = ... }], a record no root holds. *)
      Tensor.unsafe_reinitialize ();
      let _, schemes = Tensor.Scheme.traced_by (fun () -> Nn.batch_norm1d ~label:[ "bn" ] ()) in
      p_all ~min:2 "both presence patterns' schemes have the normalized result" schemes ~f:(fun s ->
          Tensor.Scheme.num_results s = 1));
  Verdict.case "a function without a () stage" (fun () ->
      Tensor.unsafe_reinitialize ();
      let _, schemes = traced (fun () -> ignore (doubled (input "x" [ 3 ]) : Tensor.t)) in
      p "its first application traces its scheme"
        (List.equal String.equal (labels schemes) [ "doubled" ]);
      let checks, traces =
        checks_and_traces (fun () -> ignore (doubled (input "x2" [ 5 ]) : Tensor.t))
      in
      p "the next one is checked against it, tracing none" (checks = 1 && traces = 0))
