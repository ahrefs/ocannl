(* gh-ocannl-404: shape schemes. [Tensor.Scheme.trace] infers a tensor function's shape constraints
   once, under a rewind; [Tensor.Scheme.instantiate] adds them, renamed, for an application to given
   argument and result shapes, without running the function.

   The oracle is direct construction: the same function applied inline, in a session of its own, to
   arguments of the same shapes, closes every argument and result to some dims (or refuses), and the
   scheme's application must close them to the same dims (or refuse). Each part of the mechanism is
   then disabled in turn ([Tensor.For_testing]'s sabotages), and the paired claim must fail -- a
   claim that passes with its part disabled pins nothing. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let outcome f =
  match f () with
  | (_ : 'a) -> Ok ()
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg
  | exception Utils.User_error msg -> Error msg

let raises ~substrings f =
  match outcome f with
  | Ok () -> false
  | Error msg ->
      Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
      (not (List.is_empty substrings))
      && List.for_all substrings ~f:(fun substring -> String.is_substring msg ~substring)

let succeeds f =
  match outcome f with
  | Ok () -> true
  | Error msg ->
      Stdio.eprintf "(not part of the golden) unexpectedly refused: %s\n%!" msg;
      false

(* A leaf of the given dims: an application's argument. *)
let leaf ?(batch = []) ?(input = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~input_dims:input
    ~output_dims:output ()

let result_shape name = Shape.make ~debug_name:name ~id:(-1) ()
let dims (sh : Shape.t) = Array.to_list (Shape.to_dims sh)

let dims_str ds =
  match ds with
  | Error _ -> "refused"
  | Ok ds ->
      String.concat ~sep:" "
        (List.map ds ~f:(fun d -> String.concat ~sep:"x" (List.map d ~f:Int.to_string)))

(* What closing a list of shapes gives: every shape's dims, or a refusal. *)
let close_all shapes =
  match List.map shapes ~f:dims with
  | ds -> Ok ds
  | exception Row.Shape_error (msg, _) ->
      Stdio.eprintf "(not part of the golden) closure refused: %s\n%!" msg;
      Error msg

let same_closure name ~inline ~scheme =
  Stdio.printf "%s: inline %s, scheme %s\n" name (dims_str inline) (dims_str scheme);
  p
    (name ^ ": the scheme's application closes as the inline one does")
    (match (inline, scheme) with
    | Ok a, Ok b -> List.equal (List.equal Int.equal) a b
    | Error _, Error _ -> true
    | _ -> false)

(* The functions under test. *)

let calls = ref 0

(* Pointwise, a reduction, a permute and an internal constant. *)
let%op body1 x =
  Int.incr calls;
  let h = (x *. x) + 0.5 in
  let r = h ++ "...|...->ab => ...|...->b a" in
  r ++ "...|...->ba => ...|...->b"

let f1 = function [ x ] -> [ body1 x ] | _ -> assert false

(* Inline: build the arguments, apply, close arguments and results. *)
let inline f args =
  Tensor.unsafe_reinitialize ();
  let args = List.map args ~f:(fun mk -> mk ()) in
  match f args with
  | results -> close_all (List.map (args @ results) ~f:(fun t -> t.Tensor.shape))
  | exception Row.Shape_error (msg, _) ->
      Stdio.eprintf "(not part of the golden) inline refused: %s\n%!" msg;
      Error msg

let instantiate_closed ?sabotage s args =
  let args = List.map args ~f:(fun mk -> mk ()) in
  let results =
    List.init (Tensor.Scheme.num_results s) ~f:(fun i -> result_shape ("res" ^ Int.to_string i))
  in
  let shapes = List.map args ~f:(fun t -> t.Tensor.shape) in
  match
    match sabotage with
    | None -> Tensor.Scheme.instantiate s ~args:shapes ~results
    | Some sabotage -> Tensor.For_testing.instantiate_scheme ~sabotage s ~args:shapes ~results
  with
  | () -> close_all (shapes @ results)
  | exception Row.Shape_error (msg, _) ->
      Stdio.eprintf "(not part of the golden) instantiation refused: %s\n%!" msg;
      Error msg

let shape_a = [ (fun () -> leaf ~batch:[ 4 ] "xa" [ 2; 3 ]) ]
let shape_b = [ (fun () -> leaf "xb" [ 5; 6 ]) ]

let () =
  Verdict.case "two shapes against direct construction" (fun () ->
      let inline_a = inline f1 shape_a and inline_b = inline f1 shape_b in
      Tensor.unsafe_reinitialize ();
      calls := 0;
      let s = Tensor.Scheme.trace ~label:"body1" 1 f1 in
      let sa = instantiate_closed s shape_a in
      let sb = instantiate_closed s shape_b in
      same_closure "shape A, one closure" ~inline:inline_a ~scheme:sa;
      same_closure "shape B, a later closure" ~inline:inline_b ~scheme:sb;
      p "the function ran once, at the trace" (!calls = 1);
      (* Both applications before one closure. *)
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"body1" 1 f1 in
      let args_a = List.map shape_a ~f:(fun mk -> mk ())
      and args_b = List.map shape_b ~f:(fun mk -> mk ()) in
      let ra = result_shape "ra" and rb = result_shape "rb" in
      Tensor.Scheme.instantiate s ~args:(List.map args_a ~f:(fun t -> t.shape)) ~results:[ ra ];
      Tensor.Scheme.instantiate s ~args:(List.map args_b ~f:(fun t -> t.shape)) ~results:[ rb ];
      let both = close_all [ ra; rb ] in
      p "both applications before one closure close as inline"
        (match (inline_a, inline_b, both) with
        | Ok a, Ok b, Ok [ da; db ] ->
            List.equal Int.equal (List.last_exn a) da && List.equal Int.equal (List.last_exn b) db
        | _ -> false);
      (* Negative control: without generalization, two applications before one closure share the
         trace's variables, so the second shape conflicts with the first. *)
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"body1" 1 f1 in
      let shapes_of args = List.map args ~f:(fun mk -> (mk ()).Tensor.shape) in
      let control () =
        Tensor.For_testing.instantiate_scheme ~sabotage:No_generalization s
          ~args:(shapes_of shape_a)
          ~results:[ result_shape "ca" ];
        Tensor.For_testing.instantiate_scheme ~sabotage:No_generalization s
          ~args:(shapes_of shape_b)
          ~results:[ result_shape "cb" ]
      in
      p "control (no generalization): the second application conflicts with the first"
        (raises ~substrings:[ "applying the shape scheme body1" ] control))

(* {2 Errors at the definition} *)

let c3 () = leaf "c3" [ 3 ]
let c4 () = leaf "c4" [ 4 ]

let () =
  Verdict.case "a shape error inside the function raises at the trace" (fun () ->
      Tensor.unsafe_reinitialize ();
      let before = Tensor.checkpoint ~label:"before the failing trace" () in
      let conflicting = function
        | [ x ] ->
            let y = NTDSL.einsum "i; i => i" (c3 ()) (c4 ()) () in
            [ NTDSL.O.( + ) x y ]
        | _ -> assert false
      in
      p "a body conflict independent of the arguments raises at the trace, naming the scheme"
        (raises ~substrings:[ "tracing the shape scheme conflicting" ] (fun () ->
             Tensor.Scheme.trace ~label:"conflicting" 1 conflicting));
      p_empty "the failed trace leaves the session as it was"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences before);
      p "the session is usable after the failed trace"
        (succeeds (fun () -> ignore (dims (leaf "after" [ 2 ]).shape : int list)));
      (* A trailing [set_dim] pushes its constraint without solving it: only the end-of-trace check
         sees the contradiction. *)
      let trailing = function
        | [ x ] ->
            let a = Shape.get_variable_ref "a" in
            let y = NTDSL.einsum1 ~capture_dims:[ a ] "...|...->a => ...|...->a" x () in
            Shape.set_dim a 3;
            [ NTDSL.einsum "...|...->a; ...|...->a => ...|...->a" y (c4 ()) () ]
        | _ -> assert false
      in
      Tensor.unsafe_reinitialize ();
      p "a trailing set_dim contradiction raises at the trace"
        (raises ~substrings:[ "tracing the shape scheme trailing" ] (fun () ->
             Tensor.Scheme.trace ~label:"trailing" 1 trailing));
      Tensor.unsafe_reinitialize ();
      p "control (no end-of-trace check): the trailing contradiction escapes the trace"
        (succeeds (fun () ->
             Tensor.For_testing.trace_scheme ~sabotage:No_end_validation ~label:"trailing" 1
               trailing));
      Tensor.unsafe_reinitialize ();
      let forcing = function
        | [ x ] ->
            let y = TDSL.O.relu x in
            ignore (dims y.shape : int list);
            [ y ]
        | _ -> assert false
      in
      p "a function that forces dims is refused before anything closes"
        (raises ~substrings:[ "while tracing the shape scheme forcing" ] (fun () ->
             Tensor.Scheme.trace ~label:"forcing" 1 forcing));
      Tensor.unsafe_reinitialize ();
      let before = Tensor.checkpoint ~label:"before a successful trace" () in
      let _s = Tensor.Scheme.trace ~label:"body1" 1 f1 in
      p_empty "a successful trace leaves the session as it was"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences before));
  Verdict.case "an argument-dependent error raises at the application" (fun () ->
      Tensor.unsafe_reinitialize ();
      let needs3 = function
        | [ x ] -> [ NTDSL.einsum "...|...->i; ...|...->i => ...|...->i" x (c3 ()) () ]
        | _ -> assert false
      in
      calls := 0;
      let counted args =
        Int.incr calls;
        needs3 args
      in
      let s = Tensor.Scheme.trace ~label:"needs3" 1 counted in
      let x = leaf "x4" [ 4 ] in
      p "an argument of the wrong size raises at the application, naming scheme and application"
        (raises ~substrings:[ "applying the shape scheme needs3 to the x4 application" ] (fun () ->
             Tensor.Scheme.instantiate ~application:"the x4 application" s ~args:[ x.shape ]
               ~results:[ result_shape "r" ]));
      p "the function did not run at the application" (!calls = 1);
      p "inline, the same argument is refused too"
        (Result.is_error (inline needs3 [ (fun () -> leaf "x4" [ 4 ]) ])))

(* {2 Shared parameters stay fixed} *)

(* [w]'s output axis is given, its input axis is inferred from the first application. *)
let%op layer () x = { w; o = [ 4 ] } * x

let shared_case () =
  Tensor.unsafe_reinitialize ();
  let lay = layer () in
  let s =
    Tensor.Scheme.trace ~label:"layer" 1 (function [ x ] -> [ lay x ] | _ -> assert false)
  in
  let apply ?sabotage x =
    let r = result_shape "y" in
    (match sabotage with
    | None -> Tensor.Scheme.instantiate s ~args:[ x.Tensor.shape ] ~results:[ r ]
    | Some sabotage ->
        Tensor.For_testing.instantiate_scheme ~sabotage s ~args:[ x.Tensor.shape ] ~results:[ r ]);
    r
  in
  apply

let () =
  Verdict.case "a shared parameter's dims stay fixed across applications" (fun () ->
      List.iter [ false; true ] ~f:(fun close_between ->
          let when_ = if close_between then "across a closure" else "before a closure" in
          (* Inline: two batches at feature 3, then feature 5. *)
          Tensor.unsafe_reinitialize ();
          let lay = layer () in
          let x2 = leaf ~batch:[ 2 ] "x2" [ 3 ] and x5 = leaf ~batch:[ 5 ] "x5" [ 3 ] in
          let y2 = lay x2 in
          if close_between then ignore (close_all [ y2.shape ]);
          let y5 = lay x5 in
          let inline_all = close_all [ y2.shape; y5.shape ] in
          let inline_conflict =
            Result.is_error
              (outcome (fun () ->
                   let y = lay (leaf "x_f5" [ 5 ]) in
                   ignore (dims y.shape : int list)))
          in
          let apply = shared_case () in
          let x2 = leaf ~batch:[ 2 ] "x2" [ 3 ] and x5 = leaf ~batch:[ 5 ] "x5" [ 3 ] in
          let r2 = apply x2 in
          if close_between then ignore (close_all [ r2 ]);
          let r5 = apply x5 in
          same_closure ("two batches " ^ when_) ~inline:inline_all ~scheme:(close_all [ r2; r5 ]);
          p ("inline, a conflicting feature size is refused " ^ when_) inline_conflict;
          p
            ("a conflicting feature size is refused at the application " ^ when_)
            (raises ~substrings:[ "applying the shape scheme layer" ] (fun () ->
                 ignore (apply (leaf "x_f5" [ 5 ]) : Shape.t)));
          (* Negative control: severed from the shared parameter, the conflict goes unnoticed. *)
          let apply = shared_case () in
          let r2 = apply (leaf ~batch:[ 2 ] "x2" [ 3 ]) in
          if close_between then ignore (close_all [ r2 ]);
          p
            ("control (severed): the conflicting feature size is no longer refused " ^ when_)
            (succeeds (fun () ->
                 ignore (apply ~sabotage:Sever_shared (leaf "x_f5" [ 5 ]) : Shape.t)))))

(* {2 Results} *)

let () =
  Verdict.case "several results: repeated, an argument, a pre-existing tensor" (fun () ->
      let mk_c () = leaf "c" [ 7 ] in
      let run_inline () =
        Tensor.unsafe_reinitialize ();
        let c = mk_c () in
        let x = leaf ~batch:[ 2 ] "x" [ 3 ] and y = leaf "y" [ 3 ] in
        let%op sum = x + y in
        close_all [ x.shape; y.shape; sum.shape; sum.shape; x.shape; c.shape ]
      in
      let inline_r = run_inline () in
      Tensor.unsafe_reinitialize ();
      let c = mk_c () in
      let multi = function
        | [ x; y ] ->
            let%op sum = x + y in
            [ sum; sum; x; c ]
        | _ -> assert false
      in
      let s = Tensor.Scheme.trace ~label:"multi" 2 multi in
      let x = leaf ~batch:[ 2 ] "x" [ 3 ] and y = leaf "y" [ 3 ] in
      let rs = List.init 4 ~f:(fun i -> result_shape ("r" ^ Int.to_string i)) in
      Tensor.Scheme.instantiate s ~args:[ x.shape; y.shape ] ~results:rs;
      same_closure "four results" ~inline:inline_r ~scheme:(close_all (x.shape :: y.shape :: rs));
      p "arity is checked before anything is written"
        (raises ~substrings:[ "the scheme has 2 arguments, the application gives 1" ] (fun () ->
             Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:rs)))

(* {2 Closure semantics carried by the scheme} *)

let () =
  Verdict.case "closure defaults and policy marks match inline" (fun () ->
      let x () = leaf ~batch:[ 2 ] "x" [ 3 ] in
      let compare name f =
        let i = inline f [ x ] in
        Tensor.unsafe_reinitialize ();
        let s = Tensor.Scheme.trace ~label:name 1 f in
        same_closure name ~inline:i ~scheme:(instantiate_closed s [ x ])
      in
      (* A [stretch] resolves at its use; a free leaf's axis closes to 1 inside a result; an
         intermediate reduction. *)
      compare "stretch" (function
        | [ x ] -> [ NTDSL.O.( + ) x (NTDSL.O.stretch (NTDSL.number 1.)) ]
        | _ -> assert false);
      compare "a free axis in the result" (function
        | [ x ] ->
            let t = NTDSL.term ~label:[ "t" ] ~fetch_op:(Constant 2.) () in
            let g = NTDSL.O.relu (NTDSL.O.( + ) t t) in
            [ NTDSL.einsum "...|...->b; ...|...->a => ...|...->ba" x g () ]
        | _ -> assert false);
      compare "an intermediate reduction" (function
        | [ x ] ->
            let u = NTDSL.term ~label:[ "u" ] ~fetch_op:(Constant 2.) () in
            let h = NTDSL.einsum1 "...|...->a => ...|...->a" (NTDSL.O.( + ) x u) () in
            [ NTDSL.einsum1 "...|...->a => ...|...->0" h () ]
        | _ -> assert false);
      (* An unsized concatenation component: closure refuses to guess it, because the trace marked
         it -- a mark the rewind takes back and the scheme re-adds. *)
      let concat = function
        | [ x ] -> [ NTDSL.einsum1 "...|...->p^o => ...|...->o" x () ]
        | _ -> assert false
      in
      compare "an unsized concatenation component" concat;
      p "inline, the unsized concatenation component is refused at the closure"
        (Result.is_error (inline concat [ x ]));
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"concat" 1 concat in
      p "control (no marks): the scheme's application no longer refuses to guess the component"
        (match instantiate_closed ~sabotage:No_marks s [ x ] with
        | Ok _ -> true
        | Error msg -> not (String.is_substring msg ~substring:"concatenation component")))

(* An intermediate's row is finalized by the closure as an update step's is: an unsized
   concatenation component off the function's boundary is refused inline, and must be by the
   scheme. *)
let () =
  Verdict.case "an intermediate's rows are finalized" (fun () ->
      let x () = leaf ~batch:[ 2 ] "x" [ 3 ] in
      let intermediate = function
        | [ x ] ->
            ignore (NTDSL.einsum1 "...|...->a => ...|...->p^a" x () : Tensor.t);
            [ x ]
        | _ -> assert false
      in
      let i = inline intermediate [ x ] in
      p "inline, the intermediate's unsized component is refused at the closure" (Result.is_error i);
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"intermediate" 1 intermediate in
      same_closure "an unsized component off the boundary" ~inline:i
        ~scheme:(instantiate_closed s [ x ]);
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"intermediate" 1 intermediate in
      p "control (no internal rows): the scheme's application accepts it instead"
        (Result.is_ok (instantiate_closed ~sabotage:No_internal_rows s [ x ])))

(* A scheme applied inside a traced function: the outer scheme keeps the inner one's anchors. *)
let () =
  Verdict.case "a scheme applied inside a trace keeps its shared parameter" (fun () ->
      Tensor.unsafe_reinitialize ();
      let lay = layer () in
      let inner =
        Tensor.Scheme.trace ~label:"inner" 1 (function [ x ] -> [ lay x ] | _ -> assert false)
      in
      let outer =
        Tensor.Scheme.trace ~label:"outer" 1 (function
          | [ x ] ->
              let y = Tensor.term ~grad_spec:Prohibit_grad ~label:[ "y" ] () in
              Tensor.Scheme.instantiate inner ~args:[ x.shape ] ~results:[ y.shape ];
              [ NTDSL.O.relu y ]
          | _ -> assert false)
      in
      let apply x =
        let r = result_shape "r" in
        Tensor.Scheme.instantiate outer ~args:[ x.Tensor.shape ] ~results:[ r ];
        r
      in
      let r2 = apply (leaf ~batch:[ 2 ] "x2" [ 3 ]) in
      let first = close_all [ r2 ] in
      p "the composed scheme applies before a closure"
        (Result.equal (List.equal (List.equal Int.equal)) String.equal first (Ok [ [ 2; 4 ] ]));
      p "after the closure, the composed scheme applies again"
        (succeeds (fun () ->
             let r5 = apply (leaf ~batch:[ 5 ] "x5" [ 3 ]) in
             ignore (close_all [ r5 ])));
      p "after the closure, a conflicting feature size is refused"
        (raises ~substrings:[ "applying the shape scheme outer" ] (fun () ->
             ignore (apply (leaf "x_f5" [ 5 ]) : Shape.t))))

(* {2 Capture references} *)

let solved (r : Shape.delayed_var_ref) = r.var_ref.solved_dim

let external_ r = function
  | [ x ] -> [ NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" x () ]
  | _ -> assert false

(* Captures [t]'s whole output row as the row variable of [r]. *)
let capture_row (r : Shape.delayed_var_ref) t =
  let l = r.var_ref.ref_label in
  NTDSL.einsum1 ~capture_dims:[ r ] (Printf.sprintf "..%s.. => ..%s.." l l) t ()

let instantiate ?sabotage s args =
  let results =
    List.init (Tensor.Scheme.num_results s) ~f:(fun i -> result_shape ("res" ^ Int.to_string i))
  in
  let args = List.map args ~f:(fun t -> t.Tensor.shape) in
  match sabotage with
  | None -> Tensor.Scheme.instantiate s ~args ~results
  | Some sabotage -> Tensor.For_testing.instantiate_scheme ~sabotage s ~args ~results

let () =
  Verdict.case "capture references" (fun () ->
      let local = function
        | [ x ] ->
            let b = Shape.get_variable_ref "b" in
            [ NTDSL.einsum1 ~capture_dims:[ b ] "...|...->ab => ...|...->a" x () ]
        | _ -> assert false
      in
      let x () = leaf "x" [ 2; 5 ] in
      let i = inline local [ x ] in
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"local" 1 local in
      same_closure "a reference made inside the function" ~inline:i
        ~scheme:(instantiate_closed s [ x ]);
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let s = Tensor.Scheme.trace ~label:"external" 1 (external_ r) in
      Stdio.printf "%s\n" (Tensor.Scheme.to_string s);
      p "the trace leaves the outside reference unbound"
        (match r.var with `Not_set_yet -> true | `Dim _ | `Row _ -> false);
      same_closure "a reference made outside the function" ~inline:i
        ~scheme:(instantiate_closed s [ x ]);
      p "the instance binds the reference, and the closure publishes its solution"
        (Option.equal Int.equal (solved r) (Some 5));
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let s = Tensor.Scheme.trace ~label:"external" 1 (external_ r) in
      ignore (instantiate_closed ~sabotage:No_ref_inputs s [ x ] : (int list list, string) Result.t);
      p "control (no reference inputs): the instance leaves the reference unsolved"
        (Option.is_none (solved r)))

(* An outside reference is a scheme input (gh-ocannl-1791): each application relates it to the
   reference's state then -- bound in open work, or solved across a closure -- as the body's own
   binding would. The scheme is traced while the reference is unbound, so only the application can
   see what a later application of the body bound it to. *)
let () =
  Verdict.case "capture references: bound per application" (fun () ->
      let conflicting ?sabotage ~close () =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "b" in
        let s = Tensor.Scheme.trace ~label:"external" 1 (external_ r) in
        (* The body applied inline binds the reference: in open work, or solved by a closure. *)
        let y = List.hd_exn (external_ r [ leaf "x1" [ 2; 5 ] ]) in
        if close then ignore (dims y.Tensor.shape : int list);
        let fitting = outcome (fun () -> instantiate ?sabotage s [ leaf "ok" [ 3; 5 ] ]) in
        let bad = outcome (fun () -> instantiate ?sabotage s [ leaf "bad" [ 3; 7 ] ]) in
        (fitting, bad)
      in
      List.iter [ false; true ] ~f:(fun close ->
          let when_ = if close then "across a closure" else "before a closure" in
          let fitting, bad = conflicting ~close () in
          p
            ("an application fitting the reference's binding succeeds " ^ when_)
            (Result.is_ok fitting);
          p
            ("a conflicting application is refused at the application " ^ when_)
            (match bad with
            | Error msg ->
                Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
                String.is_substring msg ~substring:"applying the shape scheme external to"
            | Ok () -> false);
          let _, bad = conflicting ~sabotage:No_ref_inputs ~close () in
          p
            ("control (no reference inputs): the conflicting application is accepted " ^ when_)
            (Result.is_ok bad)))

(* A row reference: rows, not products, are related -- and a dimension bound outside meets a row
   bound inside by its product, as two operations sharing the reference do. Across a closure the
   reference carries its row (gh-ocannl-1689), so the application relates that row; a scheme traced
   after the closure still reads the reference as an input, not as the row it carries. *)
let () =
  Verdict.case "capture references: rows and mixed kinds" (fun () ->
      let row_body r = function [ x ] -> [ capture_row r x ] | _ -> assert false in
      let rows ?sabotage ?(close = false) ?(trace_after = false) arg =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "q" in
        let trace () = Tensor.Scheme.trace ~label:"row_capture" 1 (row_body r) in
        let s = if trace_after then None else Some (trace ()) in
        let y = capture_row r (leaf "x1" [ 2; 6 ]) in
        if close then ignore (dims y.Tensor.shape : int list);
        let s = match s with Some s -> s | None -> trace () in
        outcome (fun () -> instantiate ?sabotage s [ arg () ])
      in
      List.iter
        [
          (false, false, ""); (true, false, " across a closure"); (true, true, ", traced after it");
        ]
        ~f:(fun (close, trace_after, when_) ->
          p
            ("a row reference bound to 2x6" ^ when_ ^ ": an application to 2x6 succeeds")
            (Result.is_ok (rows ~close ~trace_after (fun () -> leaf "same" [ 2; 6 ])));
          p
            ("an application to 3x4" ^ when_
           ^ ", of the same product, is refused: the rows are related")
            (Result.is_error (rows ~close ~trace_after (fun () -> leaf "other" [ 3; 4 ])));
          p
            ("control (no reference inputs)" ^ when_ ^ ": the application to 3x4 is accepted")
            (Result.is_ok
               (rows ~sabotage:No_ref_inputs ~close ~trace_after (fun () -> leaf "other" [ 3; 4 ]))));
      let mixed arg =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "b" in
        let s = Tensor.Scheme.trace ~label:"external" 1 (external_ r) in
        ignore (capture_row r (leaf "x1" [ 2; 6 ]) : Tensor.t);
        outcome (fun () -> instantiate s [ arg () ])
      in
      p "a reference bound outside to a row of 12 elements: a dimension of 12 inside fits"
        (Result.is_ok (mixed (fun () -> leaf "twelve" [ 3; 12 ])));
      p "and a dimension of 6 does not" (Result.is_error (mixed (fun () -> leaf "six" [ 3; 6 ]))))

(* One input however often the body binds the reference; a scheme applied inside another trace
   passes its inputs on to the outer scheme. *)
let () =
  Verdict.case "capture references: repeated and nested" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let twice = function
        | [ x ] ->
            let y = NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->a" x () in
            [ y; NTDSL.einsum1 ~capture_dims:[ r ] "...|...->ab => ...|...->b" x () ]
        | _ -> assert false
      in
      let s = Tensor.Scheme.trace ~label:"twice" 1 twice in
      Stdio.printf "%s\n" (Tensor.Scheme.to_string s);
      Shape.set_dim r 5;
      p "a body binding the reference twice: an application fitting it succeeds"
        (Result.is_ok (outcome (fun () -> instantiate s [ leaf "ok" [ 5; 5 ] ])));
      p "and a conflicting one is refused"
        (Result.is_error (outcome (fun () -> instantiate s [ leaf "bad" [ 5; 7 ] ])));
      let nested ?sabotage () =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "b" in
        let inner = Tensor.Scheme.trace ~label:"external" 1 (external_ r) in
        let outer = function
          | [ x ] ->
              let y = NTDSL.einsum1 "...|...->ab => ...|...->a" x () in
              (match sabotage with
              | None -> Tensor.Scheme.instantiate inner ~args:[ x.shape ] ~results:[ y.shape ]
              | Some sabotage ->
                  Tensor.For_testing.instantiate_scheme ~sabotage inner ~args:[ x.shape ]
                    ~results:[ y.shape ]);
              [ y ]
          | _ -> assert false
        in
        let s = Tensor.Scheme.trace ~label:"outer" 1 outer in
        Shape.set_dim r 5;
        outcome (fun () -> instantiate s [ leaf "bad" [ 3; 7 ] ])
      in
      p
        "an outer scheme applying the inner one carries its input: a conflicting application is \
         refused"
        (Result.is_error (nested ()));
      p "control (the inner application drops its input): the outer scheme accepts it"
        (Result.is_ok (nested ~sabotage:No_ref_inputs ())))

(* What stays refused: an outside reference set or related, not bound, inside the function. *)
let () =
  Verdict.case "capture references: what stays refused" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "b" in
      let sets = function
        | [ x ] ->
            Shape.set_dim r 5;
            external_ r [ x ]
        | _ -> assert false
      in
      p "a function setting a reference made outside it is refused at the trace, naming the call"
        (raises ~substrings:[ "tracing the shape scheme sets"; "Shape.set_dim"; "made outside it" ]
           (fun () -> Tensor.Scheme.trace ~label:"sets" 1 sets));
      p "control: the same function applied inline sets the reference"
        (succeeds (fun () -> ignore (sets [ leaf "x" [ 2; 5 ] ] : Tensor.t list))))

(* One reference bound both as a row and as a dimension inside the function (gh-ocannl-1890): the
   product tie between the two is a constant, so the trace carries it. The row of [x]'s output axes
   and its last axis are then one product: only a leading axis of 1 fits. *)
let mixed_row_first r = function
  | [ x ] ->
      ignore (capture_row r x : Tensor.t);
      external_ r [ x ]
  | _ -> assert false

let mixed_dim_first r = function
  | [ x ] ->
      let ys = external_ r [ x ] in
      ignore (capture_row r x : Tensor.t);
      ys
  | _ -> assert false

let mixed_inside args = mixed_row_first (Shape.get_variable_ref "b") args

let () =
  Verdict.case "capture references: one bound both as a row and as a dimension" (fun () ->
      let inline_mixed body arg =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "b" in
        let x = arg () in
        match body r [ x ] with
        | results -> close_all (List.map (x :: results) ~f:(fun t -> t.Tensor.shape))
        | exception Row.Shape_error (msg, _) ->
            Stdio.eprintf "(not part of the golden) inline refused: %s\n%!" msg;
            Error msg
      in
      let schemed_mixed body arg =
        Tensor.unsafe_reinitialize ();
        let r = Shape.get_variable_ref "b" in
        match Tensor.Scheme.trace ~label:"mixed" 1 (body r) with
        | s -> instantiate_closed s [ arg ]
        | exception Row.Shape_error (msg, _) ->
            Stdio.eprintf "(not part of the golden) trace refused: %s\n%!" msg;
            Error ("trace: " ^ msg)
      in
      let fits () = leaf "fits" [ 1; 6 ] and conflicts () = leaf "conflicts" [ 2; 6 ] in
      List.iter
        [
          ("made outside, row first", mixed_row_first);
          ("made outside, dimension first", mixed_dim_first);
          ("made inside", fun _ -> mixed_inside);
        ]
        ~f:(fun (name, body) ->
          let scheme = schemed_mixed body fits in
          p
            (name ^ ": the trace is accepted and a fitting application closes")
            (Result.is_ok scheme);
          same_closure (name ^ ", fitting") ~inline:(inline_mixed body fits) ~scheme;
          let inline = inline_mixed body conflicts in
          p (name ^ ": inline, a conflicting product is refused") (Result.is_error inline);
          same_closure (name ^ ", conflicting") ~inline ~scheme:(schemed_mixed body conflicts)))

(* {2 Rollback and delayed diagnostics} *)

let () =
  Verdict.case "an application inside a checkpoint rewinds" (fun () ->
      Tensor.unsafe_reinitialize ();
      let s = Tensor.Scheme.trace ~label:"body1" 1 f1 in
      let x = leaf "x" [ 2; 3 ] in
      let ck = Tensor.checkpoint ~label:"before the application" () in
      let r = result_shape "r" in
      Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:[ r ];
      p "the application changes the session"
        (not (List.is_empty (Tensor.For_testing.checkpoint_differences ck)));
      Tensor.rewind ck;
      p_empty "after the rewind, the session is as at the checkpoint"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p "the rewound application's result shape is a discarded handle"
        (Option.is_some r.discarded_by);
      let r2 = result_shape "r2" in
      Tensor.Scheme.instantiate s ~args:[ x.shape ] ~results:[ r2 ];
      same_closure "applying again after the rewind"
        ~inline:(inline f1 [ (fun () -> leaf "x" [ 2; 3 ]) ])
        ~scheme:(close_all [ x.shape; r2 ]));
  Verdict.case "a closure-time failure names the application" (fun () ->
      Tensor.unsafe_reinitialize ();
      let s =
        Tensor.Scheme.trace ~label:"concat" 1 (function
          | [ x ] -> [ NTDSL.einsum1 "...|...->p^o => ...|...->o" x () ]
          | _ -> assert false)
      in
      let x = leaf ~batch:[ 2 ] "x" [ 3 ] in
      let r = result_shape "r" in
      Tensor.Scheme.instantiate ~application:"the probe application" s ~args:[ x.shape ]
        ~results:[ r ];
      p "the closure's error trace carries the scheme's and the application's origin"
        (match Shape.to_dims r with
        | _ -> false
        | exception (Row.Shape_error _ as e) ->
            String.is_substring
              (Sexp.to_string (sexp_of_exn e))
              ~substring:"the shape scheme concat applied to the probe application"))

(* gh-ocannl-1753: [Tensor.Scheme.to_string] names variables by first appearance, nested ones in a
   compound dimension included, so the text does not depend on the id counters: a trace's ids come
   from the reserved range, which every trace advances. *)
let () =
  Verdict.case "a printed scheme names every variable" (fun () ->
      Tensor.unsafe_reinitialize ();
      let concat = function
        | [ x ] -> [ NTDSL.einsum1 "...|...->p^o => ...|...->o" x () ]
        | _ -> assert false
      in
      let print () = Tensor.Scheme.to_string (Tensor.Scheme.trace ~label:"concat" 1 concat) in
      let first = print () in
      let (_ : string) = print () in
      let third = print () in
      Stdio.eprintf "(not part of the golden) the printed scheme:\n%s\n%!" first;
      p "the argument's concatenated axis is printed" (String.is_substring first ~substring:"^");
      p "no raw variable id is printed" (not (String.mem first '$'));
      p "the text is the same after more traces advanced the ids" (String.equal first third))
