(* Tutorial: reusable tensor programs (gh-ocannl-1753).

   A [%op] function is a tensor program you define once and apply many times. Since v1.2 each such
   function gets a *shape scheme*: the shape constraints its body imposes, traced once and checked
   at every application. This tutorial shows what that buys a newcomer, in three steps:

   1. a tensor function defined once and applied at two shapes, with its scheme printed; 2. a shape
   mistake inside a tensor function, reported at its definition or at its first use, and naming the
   function; 3. a notebook- or REPL-style session in which a failed construction is rewound to a
   checkpoint, and the session continues to inference.

   The [%expect] blocks are this file's real output, checked by [dune build
   @test/operations/runtest-reusable_tensor_programs]. The reference for the mechanism is
   docs/syntax_extensions.md, "Automatic shape schemes", and docs/shape_inference.md, "Shape
   schemes"; the checkpoint API is in tensor/tensor.mli ([Tensor.checkpoint] and friends). *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

(* {1 The tensor functions}

   [mlp_layer ~hid ()] builds a layer: the [()] application creates its parameters [w] and [b]
   (lifted inline declarations), and returns the function [fun x -> ...] that uses them. The output
   size is given ([hid]); the input size of [w] is left to inference: the first application fixes
   it. *)
let%op mlp_layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })

(* A block that builds its sub-layer at [()], then returns the function. Its residual connection
   needs the layer's output size to equal the argument's feature size. *)
let%op residual_block ~hid () =
  let layer = mlp_layer ~hid () in
  fun x -> x + layer x

(* A two-layer network with a typo: the second bias is declared with the hidden size instead of the
   output size. No argument can make this right. *)
let%op two_layer ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }

(* The same function with its scheme opted out, for contrast: [%op] expands it as it did before
   schemes. *)
let%op two_layer_unchecked ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }
[@@ocannl.no_scheme]

(* {1 Helpers} *)

(* An input of the given batch and feature (output) sizes, with no values: shape inference only. *)
let input ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

(* [Tensor.Scheme.traced_by build] runs [build] and returns its result with the schemes the closures
   built in it traced -- the ones their applications are checked against -- and
   [Tensor.Scheme.to_string] prints them. *)
let print_traced build =
  let f, schemes = Tensor.Scheme.traced_by build in
  List.iter schemes ~f:(fun s -> Stdio.print_endline (Tensor.Scheme.to_string s));
  f

(* [Tensor.Scheme.trace] traces a one-argument function's scheme explicitly: the trace the [()]
   application ran, run again now. *)
let print_scheme ~label f =
  let s = Tensor.Scheme.trace ~label 1 (function [ x ] -> [ f x ] | _ -> assert false) in
  Stdio.print_endline (Tensor.Scheme.to_string s)

(* A shape error, printed as the toplevel prints it ([Printexc.to_string]): the message, the sizes
   in conflict, the constraint that failed, and the tensors it relates with their axis kinds.
   Variables are named by first appearance (['a], ..['b]..), and so is a tensor of a trace or a
   check ([n'1]): their ids come from a reserved range. [Shape.error_to_string_hum] gives the same
   text, and the setting [shape_error_full_sexp=true] prints the solver's raw constraints
   instead. *)
let print_shape_error exn = Stdio.print_endline (Stdlib.Printexc.to_string exn)

let attempt f =
  match f () with
  | (_ : Tensor.t) -> Stdio.print_endline "built"
  | exception (Row.Shape_error _ as exn) -> print_shape_error exn
  | exception Tensor.Session_error (msg, _) -> Stdio.print_endline ("Session_error: " ^ msg)

let shape t = Shape.to_string_hum t.Tensor.shape

(* {1 Step 1: one function, two shapes} *)

let%expect_test "a tensor function defined once, applied at two shapes" =
  Tensor.unsafe_reinitialize ();
  (* The scheme the [()] application traced reads like a type: [batch|input->output] for the
     argument, [=>], the same for the result. [..'a..] is a row variable (any number of axes),
     generalized: each application gets its own. [..'_a..] is fixed: it belongs to [w], which every
     application shares, as OCaml prints a weak type variable. The indented lines are the
     broadcasting relations, [r ⊑ o] reading "the result row r refines the operand row o" (equal, or
     o broadcasts to r). So: the result keeps the argument's batch and input axes (and [b]'s), its
     output ends in the 4 features, and the argument's output row must broadcast to [w]'s input
     row. *)
  let layer = print_traced (fun () -> mlp_layer ~hid:4 ()) in
  [%expect
    {|
    mlp_layer : ..'a..|..'b..->..'c.. => ..'d..|..'e..->..'f..,4
      ..'d.. ⊑ ..'a..
      ..'e.. ⊑ ..'b.., ..'_a..
      ..'_b.. ⊑ ..'c..
    |}];
  (* Two applications at different batch shapes: each checks the scheme, then builds as usual. *)
  let y1 = layer (input ~batch:[ 2 ] "x1" [ 3 ]) in
  let y2 = layer (input ~batch:[ 5; 7 ] "x2" [ 3 ]) in
  (* [Shape.to_string_hum] prints [batch|output] (no input axes here). *)
  Stdio.printf "y1: %s\ny2: %s\n" (shape y1) (shape y2);
  [%expect {|
    y1: 2|4
    y2: 5,7|4
    |}];
  (* The first application fixed [w]'s input size to 3. The closure keeps the scheme it traced at
     [()]: its fixed variables are [w]'s, so each check relates them to [w]'s state then. A trace
     run now says so in its text -- [w]'s input row ends in 3, and bounds the argument's output row
     from below. *)
  print_scheme ~label:"layer" layer;
  [%expect
    {|
    layer : ..'a..|..'b..->..'c.. => ..'d..|..'e..->..'f..,4
      ..'d.. ⊑ ..'a..
      ..'e.. ⊑ ..'b.., ..'_a..
      ..'_b..,3 ⊑ ..'c..
    |}]

(* {1 Step 2: a shape mistake, reported naming the function} *)

let%expect_test "a shape mistake reported at the definition or the first use" =
  Tensor.unsafe_reinitialize ();
  (* A mistake that no argument can fix is found when the function is built: the [()] application
     traces the scheme, and the trace fails, naming the function. *)
  attempt (fun () ->
      let net = two_layer ~hid:8 ~out:2 () in
      net (input "x" [ 3 ]));
  [%expect
    {|
    Shape_error: tracing the shape scheme two_layer: solved dimensions for axis: mismatch
      conflicting sizes: 2 vs 8
      failed constraint: 'a = 8
        from two_layer (output) with b2 (output): pointwise_binary
    |}];
  (* To find it at the definition, at program start, check the definition against a trial
     configuration: [Tensor.Scheme.check_definition] builds it in a window that is always rewound,
     and leaves the session as it was. On success it returns the schemes it traced, innermost
     first. *)
  let check name f =
    match Tensor.Scheme.check_definition ~name f with
    | traced ->
        Stdio.printf "%s: ok, traced %s\n" name
          (String.concat ~sep:", " (List.map traced ~f:Tensor.Scheme.label))
    | exception (Row.Shape_error _ as exn) -> print_shape_error exn
  in
  check "residual_block" (fun () -> ignore (residual_block ~hid:4 () : Tensor.t -> Tensor.t));
  check "two_layer" (fun () -> ignore (two_layer ~hid:8 ~out:2 () : Tensor.t -> Tensor.t));
  [%expect
    {|
    residual_block: ok, traced mlp_layer, residual_block
    Shape_error: checking the definition two_layer: tracing the shape scheme two_layer: solved dimensions for axis: mismatch
      conflicting sizes: 2 vs 8
      failed constraint: 'a = 8
        from two_layer (output) with b2 (output): pointwise_binary
    |}];
  (* For contrast: without its scheme, the same typo is found only when the body runs at the first
     application, and the message names neither the function nor the argument (only the label of the
     tensor computing its result does). The failure happened inside the construction, so it also
     taints the session (step 3 shows the recovery; here we just rewind to the last closure). *)
  attempt (fun () ->
      let net = two_layer_unchecked ~hid:8 ~out:2 () in
      net (input "x" [ 3 ]));
  Tensor.rewind_to_last_closure ();
  [%expect
    {|
    Shape_error: solved dimensions for axis: mismatch
      conflicting sizes: 2 vs 8
      failed constraint: 'a = 8
        from two_layer_unchecked (output) with b2 (output): pointwise_binary
    |}];
  (* A mistake that depends on the argument is found at the application, before the body runs: the
     residual connection needs 4 features, the argument has 3. *)
  let block = residual_block ~hid:4 () in
  attempt (fun () -> block (input ~batch:[ 2 ] "x" [ 3 ]));
  [%expect
    {|
    Shape_error: applying the shape scheme residual_block to x: dimension comparison for axis: mismatch
      conflicting sizes: 4 vs 3
      failed constraint: 4 ⊑ 3
        from residual_block (output) with x (output): the shape scheme residual_block applied to x
        from residual_block (output) with (its trace's solution) (output): the trace of the shape scheme residual_block
        from residual_block (output) with n'1 (output): pointwise_binary
        from w (input) with n'1 (output): compose
        from w (input) with n66 (input): pointwise_binary
        from n66 (input) with n64 (input): pointwise_binary
        from n64 (batch) with threefry4x32_light (output): Uint4x32_to_prec RESULT total elements
    |}];
  (* So is an application that contradicts what an earlier one fixed: [layer]'s first application
     made [w]'s input size 3. *)
  let layer = mlp_layer ~hid:4 () in
  attempt (fun () -> layer (input ~batch:[ 2 ] "x3" [ 3 ]));
  attempt (fun () -> layer (input ~batch:[ 2 ] "x6" [ 6 ]));
  [%expect
    {|
    built
    Shape_error: applying the shape scheme mlp_layer to x6: dimension comparison for axis: mismatch
      conflicting sizes: 3 vs 6
      failed constraint: 3 ⊑ 6
        from mlp_layer (output) with x6 (output): the shape scheme mlp_layer applied to x6
        from w (input) with n'1 (output): compose
    |}]

(* {1 Step 3: a REPL session that recovers from a failed construction} *)

let%expect_test "a failed construction rewound to a checkpoint, and the session continues" =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Each phrase below is what one would type at a toplevel or run in a notebook cell. *)
  (* The data: a batch of 2 examples with 3 features. *)
  let x =
    Tensor.term_init ~grad_spec:Prohibit_grad [| 0.; 1.; 2.; 3.; 4.; 5. |] ~label:[ "x" ]
      ~batch_dims:[ 2 ] ~output_dims:[ 3 ] ()
  in
  (* Before trying out models, take a checkpoint of the open inference session. *)
  let before_model = Tensor.checkpoint ~label:"before the model" () in
  (* A first try: the residual block is too wide for the data. The scheme refuses the application,
     but the block's parameters were created by [()] and are still in the session. *)
  attempt (fun () -> residual_block ~hid:4 () x);
  [%expect
    {|
    Shape_error: applying the shape scheme residual_block to x: dimension comparison for axis: mismatch
      conflicting sizes: 4 vs 3
      failed constraint: 4 ⊑ 3
        from residual_block (output) with x (output): the shape scheme residual_block applied to x
        from residual_block (output) with (its trace's solution) (output): the trace of the shape scheme residual_block
        from residual_block (output) with n'1 (output): pointwise_binary
        from w (input) with n'1 (output): compose
        from w (input) with n8 (input): pointwise_binary
        from n8 (input) with n6 (input): pointwise_binary
        from n6 (batch) with threefry4x32_light (output): Uint4x32_to_prec RESULT total elements
    |}];
  (* A second try, written without a tensor function, fails inside the construction: that leaves the
     session tainted, and every later construction is refused, naming the first failure. *)
  attempt (fun () -> [%op x + { bias = 0.; o = [ 5 ] }]);
  attempt (fun () -> [%op x + 1.0]);
  [%expect
    {|
    Shape_error: solved dimensions for axis: mismatch
      conflicting sizes: 3 vs 5
      failed constraint: 'a = 5
        from n16 (output) with bias (output): pointwise_binary
    Shape_error: the session is inconsistent after an earlier shape error at shape propagation for #16 n16: solved dimensions for axis: mismatch; call Tensor.rewind_to_last_closure () or Tensor.unsafe_reinitialize ()
    |}];
  (* Rewinding to the checkpoint discards everything created since -- the unused parameters and the
     failed construction -- and clears the taint. *)
  Tensor.rewind before_model;
  (* [Tensor.with_checkpoint] does both in one phrase: it rewinds only when its body raises. *)
  attempt (fun () -> Tensor.with_checkpoint (fun () -> [%op x + { bias = 0.; o = [ 5 ] }]));
  [%expect
    {|
    Shape_error: solved dimensions for axis: mismatch
      conflicting sizes: 3 vs 5
      failed constraint: 'a = 5
        from n19 (output) with bias (output): pointwise_binary
    |}];
  (* The session continues: the right model builds, compiles and runs. *)
  let block = residual_block ~hid:3 () in
  let y = block x in
  let (_ : Context.t) = Train.forward_once ctx y in
  Stdio.printf "y: %s\n" (shape y);
  [%expect {| y: 2|3 |}]
