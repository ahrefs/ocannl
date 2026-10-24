open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* Dimension captures are bound per call (gh-ocannl-1507). A capture names a dimension variable of
   one einsum application, so the ppx binds it at the top of its innermost enclosing function body,
   while inline declarations keep their lifting to the unit parameter or the definition. Before
   this, a capture was bound wherever the inline declarations were. Every call then shared one
   reference cell, which the first solution pinned, so a second input of another size failed shape
   inference, or, with both calls built before inference closed, could read the other call's size.

   Each case applies one function to inputs of two widths and checks the executed values. The
   averages are exact by construction: every input cell holds 2, so each call's average is 2 exactly
   when it divides by its own width. *)

let%op mean_last x = (x ++ "ab => a" [ "b" ]) /. dim b

(* The parameter [s] belongs to the instance and the capture [b] to the call. *)
let%op scaled_mean () x = { s = 1.0 } *. ((x ++ "ab => a" [ "b" ]) /. dim b)

let filled width =
  NTDSL.init ~l:(Printf.sprintf "x%d" width) ~prec:Ir.Ops.single ~b:[] ~o:[ 2; width ]
    ~f:(fun _ -> 2.)
    ()

let observe y =
  Train.set_materialized y.Tensor.value;
  Ir.Tnode.set_observable y.Tensor.value

let check_means ctx name ys =
  List.iter ys ~f:(fun (width, y) ->
      p_all ~min:2
        (Printf.sprintf "%s: the width-%d call averages over its own width" name width)
        (Array.to_list (Context.get_values ctx y.Tensor.value))
        ~f:(Float.equal 2.))

let () =
  Verdict.case "unit-less function, calls inferred one after the other" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "unit-less, sequential" in
      let y3 = mean_last (filled 3) in
      observe y3;
      let ctx = Train.forward_once (Context.auto ()) y3 in
      let y5 = mean_last (filled 5) in
      observe y5;
      let ctx = Train.forward_once ctx y5 in
      check_means ctx name [ (3, y3); (5, y5) ];
      Context.release ctx);
  Verdict.case "unit-less function, both calls built before inference" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "unit-less, built together" in
      let ys = [ (3, mean_last (filled 3)); (5, mean_last (filled 5)) ] in
      List.iter ys ~f:(fun (_, y) -> observe y);
      let ctx =
        List.fold ys ~init:(Context.auto ()) ~f:(fun ctx (_, y) -> Train.forward_once ctx y)
      in
      check_means ctx name ys;
      Context.release ctx);
  Verdict.case "one instance of a unit-parameter function" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "unit-parameter instance" in
      let f = scaled_mean () in
      let y3 = f (filled 3) and y5 = f (filled 5) in
      let ys = [ (3, y3); (5, y5) ] in
      List.iter ys ~f:(fun (_, y) -> observe y);
      let ctx =
        List.fold ys ~init:(Context.auto ()) ~f:(fun ctx (_, y) -> Train.forward_once ctx y)
      in
      check_means ctx name ys;
      p "unit-parameter instance: both calls use the instance's one parameter"
        (Set.length y3.params = 1 && Set.equal y3.params y5.params);
      Context.release ctx);
  (* [batch_norm1d]'s moments capture the batch axes, so one layer must accept any batch size. The
     batch's first half holds 1 and its second half 3: mean 2, population variance 1. *)
  Verdict.case "one batch_norm1d instance at two batch sizes" (fun () ->
      Tensor.unsafe_reinitialize ();
      let bn = Nn_blocks.batch_norm1d ~label:[ "bn" ] () in
      let batch n =
        NTDSL.init ~l:(Printf.sprintf "batch%d" n) ~prec:Ir.Ops.single ~b:[ n ] ~o:[ 1 ]
          ~f:(function [| r; _ |] -> if r < n / 2 then 1. else 3. | _ -> 0.)
          ()
      in
      let ys = List.map [ 2; 4 ] ~f:(fun n -> (n, bn ~train_step:(Some 0) (batch n))) in
      List.iter ys ~f:(fun (_, y) -> observe y);
      let ctx =
        List.fold ys ~init:(Context.auto ()) ~f:(fun ctx (_, y) -> Train.forward_once ctx y)
      in
      let scale = 1. /. Float.sqrt (1. +. 1e-5) in
      List.iter ys ~f:(fun (n, y) ->
          p_close ~min:n
            (Printf.sprintf "batch_norm1d: the batch-%d call normalizes with its own batch size" n)
            (Context.get_values ctx y.Tensor.value)
            (Array.init n ~f:(fun r -> if r < n / 2 then -.scale else scale))
            ~tol:(fun _ -> 1e-5));
      Context.release ctx)

(* A reference shared by hand names one dimension (gh-ocannl-1545). Where the ppx makes a fresh
   reference per call, a [Shape.get_variable_ref] passed through [~capture_dims] to several
   operations ties them within one inference: dimension variables are equated, row variables too,
   and a dimension with a row by the row's product. This holds whatever order inference runs in:
   before the fix, a reference whose dimension was still open when the next operation registered was
   silently retargeted to that operation's variable, leaving the earlier one untied, while a
   reference already solved did tie. Each case registers the operations on leaves whose shapes are
   still open, asserts the reference unsolved, and only then fixes one operand's size, by an
   independent capture set with [Shape.set_dim], reading the other operands' sizes after closure. *)

let open_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:(Constant 2.) ()
let capture ?(spec = "d=>d") d t = NTDSL.einsum1 ~capture_dims:[ d ] spec t ()

(* An independent capture fixes [t]'s axes to [sizes] (one axis per size): equality, not the
   broadcasting a pointwise use would allow. *)
let fix t sizes =
  let labels =
    List.mapi sizes ~f:(fun i _ -> String.of_char (Char.of_int_exn (Char.to_int 'p' + i)))
  in
  let refs = List.map labels ~f:Shape.get_variable_ref in
  List.iter2_exn refs sizes ~f:Shape.set_dim;
  let axes = String.concat labels in
  ignore (NTDSL.einsum1 ~capture_dims:refs (axes ^ "=>" ^ axes) t () : Tensor.t)

let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)
let product t = List.fold (dims t) ~init:1 ~f:( * )
let dims_str t = String.concat ~sep:"x" (List.map (dims t) ~f:Int.to_string)

let unsolved name (d : Shape.delayed_var_ref) =
  p
    (name ^ ": the reference is still unsolved when the operations have registered")
    (Option.is_none d.var_ref.solved_dim && Option.is_none d.var_ref.solved_sym)

let show name ts =
  List.iter ts ~f:(fun (label, t) -> Stdio.printf "%s: %s is %s\n" name label (dims_str t))

let raises_shape_error f =
  match f () with
  | () -> false
  | exception Row.Shape_error _ ->
      Tensor.rewind_to_last_closure ~reason:"gh-1545 conflict case" ();
      true

let () =
  Verdict.case "shared reference, both open, the second operand fixed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "both open, second fixed" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
      let _y1 = capture d w1 and _y2 = capture d w2 in
      unsolved name d;
      fix w2 [ 5 ];
      show name [ ("w1", w1); ("w2", w2) ];
      p (name ^ ": the first operand follows the second") (List.equal Int.equal (dims w1) [ 5 ]));
  Verdict.case "shared reference, both open, the first operand fixed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "both open, first fixed" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
      let _y1 = capture d w1 and _y2 = capture d w2 in
      unsolved name d;
      fix w1 [ 5 ];
      show name [ ("w1", w1); ("w2", w2) ];
      p (name ^ ": the second operand follows the first") (List.equal Int.equal (dims w2) [ 5 ]));
  Verdict.case "shared reference, three registrations" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "three registrations" in
      let d = Shape.get_variable_ref "d" in
      let ws = List.map [ "w1"; "w2"; "w3" ] ~f:(fun l -> (l, open_leaf l)) in
      List.iter ws ~f:(fun (_, w) -> ignore (capture d w : Tensor.t));
      unsolved name d;
      fix (snd (List.last_exn ws)) [ 4 ];
      show name ws;
      p_all ~min:3 (name ^ ": every operand follows the last") ws ~f:(fun (_, w) ->
          List.equal Int.equal (dims w) [ 4 ]));
  Verdict.case "shared reference, both open, then Shape.set_dim" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "both open, then set_dim" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
      let _y1 = capture d w1 and _y2 = capture d w2 in
      unsolved name d;
      Shape.set_dim d 5;
      show name [ ("w1", w1); ("w2", w2) ];
      p_all ~min:2 (name ^ ": set_dim reaches both operands") [ w1; w2 ] ~f:(fun w ->
          List.equal Int.equal (dims w) [ 5 ]));
  Verdict.case "shared reference, both open, conflicting sizes" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "both open, conflicting" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
      let _y1 = capture d w1 and _y2 = capture d w2 in
      unsolved name d;
      p
        (name ^ ": sizes 3 and 5 for one reference are a shape error")
        (raises_shape_error (fun () ->
             fix w1 [ 3 ];
             fix w2 [ 5 ];
             ignore (dims w1 : int list))));
  Verdict.case "shared reference, solved first" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "solved first" in
      let d = Shape.get_variable_ref "d" in
      let c5 = NTDSL.init ~l:"c5" ~prec:Ir.Ops.single ~b:[] ~o:[ 5 ] ~f:(fun _ -> 2.) () in
      let _y1 = capture d c5 in
      p
        (name ^ ": the first registration solved the reference")
        (Option.is_some d.var_ref.solved_dim);
      let w2 = open_leaf "w2" in
      let _y2 = capture d w2 in
      show name [ ("w2", w2) ];
      p (name ^ ": the open operand takes the solved size") (List.equal Int.equal (dims w2) [ 5 ]);
      let c3 = NTDSL.init ~l:"c3" ~prec:Ir.Ops.single ~b:[] ~o:[ 3 ] ~f:(fun _ -> 2.) () in
      p
        (name ^ ": a different size after the reference is solved is a shape error")
        (raises_shape_error (fun () ->
             let y3 = capture d c3 in
             ignore (dims y3 : int list))));
  Verdict.case "shared reference, reused after a closure" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "after a closure" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" in
      let y1 = capture d w1 in
      fix w1 [ 5 ];
      ignore (dims y1 : int list);
      let w2 = open_leaf "w2" in
      let _y2 = capture d w2 in
      show name [ ("w2", w2) ];
      p (name ^ ": the later operand takes the closed size") (List.equal Int.equal (dims w2) [ 5 ]));
  (* A row holding a symbolic extent has no numeric product, so its closure records neither
     [solved_dim] nor [solved_sym], and the row variable the reference was bound to belongs to the
     reset environment. The next binding is not tied to it and keeps its own shape. *)
  Verdict.case "shared reference, reused after a closure that recorded no value" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "after a valueless closure" in
      let module Idx = Ir.Indexing in
      let seq, _bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
      let e = Shape.get_variable_ref "e" in
      Shape.set_sym_dim e seq;
      let w1 = open_leaf "w1" in
      let _ = capture e ~spec:"e=>e" w1 in
      let d = Shape.get_variable_ref "d" in
      let y1 = capture ~spec:"..d.. => ..d.." d w1 in
      ignore (dims y1 : int list);
      unsolved name d;
      let w2 = open_leaf "w2" in
      let _y2 = capture ~spec:"..d.. => ..d.." d w2 in
      fix w2 [ 2; 3 ];
      show name [ ("w2", w2) ];
      p (name ^ ": the later operand keeps its own shape") (List.equal Int.equal (dims w2) [ 2; 3 ]));
  Verdict.case "shared reference, symbolic extent" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "symbolic" in
      let module Idx = Ir.Indexing in
      let check label set_before =
        let seq, _bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
        let d = Shape.get_variable_ref "d" in
        if set_before then Shape.set_sym_dim d seq;
        let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
        let _y1 = capture d w1 and _y2 = capture d w2 in
        if not set_before then (
          unsolved (name ^ ", " ^ label) d;
          Shape.set_sym_dim d seq);
        let want = Idx.symbol_ident seq.static_symbol ^ "<=6" in
        (* Reading the dims closes inference, so the printed shapes are solved. *)
        List.iter [ w1; w2 ] ~f:(fun w -> ignore (dims w : int list));
        p_all ~min:2 (Printf.sprintf "%s, %s: both operands carry the symbolic extent" name label)
          [ w1; w2 ] ~f:(fun w ->
            String.equal (Shape.to_string_hum ~style:Axis_size w.Tensor.shape) want)
      in
      check "set before the registrations" true;
      check "set after both open registrations" false);
  (* Within one inference, a row reference names one row: two row bindings are [Row_eq]-tied, in
     whichever order they resolve. A dimension and a row share only a number, the row's product.
     Across a closure only the product survives (the reference's [solved_dim]). *)
  Verdict.case "shared reference, rows" (fun () ->
      Tensor.unsafe_reinitialize ();
      let row = "..d.. => ..d.." in
      let check name ~spec1 ~spec2 ~fix_first ~want =
        let d = Shape.get_variable_ref "d" in
        let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
        let _y1 = capture ~spec:spec1 d w1 and _y2 = capture ~spec:spec2 d w2 in
        unsolved name d;
        let fixed, other = if fix_first then (w1, w2) else (w2, w1) in
        fix fixed [ 2; 3 ];
        show name [ ("w1", w1); ("w2", w2) ];
        match want with
        | `Same_row ->
            p
              (name ^ ": the other operand has the fixed operand's row")
              (List.equal Int.equal (dims other) [ 2; 3 ])
        | `Same_product ->
            p (name ^ ": the other operand has the fixed operand's product") (product other = 6)
      in
      check "rows, both open" ~spec1:row ~spec2:row ~fix_first:false ~want:`Same_row;
      check "rows, both open, first fixed" ~spec1:row ~spec2:row ~fix_first:true ~want:`Same_row;
      check "dim then row" ~spec1:"d=>d" ~spec2:row ~fix_first:false ~want:`Same_product;
      check "row then dim" ~spec1:row ~spec2:"d=>d" ~fix_first:true ~want:`Same_product);
  Verdict.case "shared reference, rows solved first" (fun () ->
      Tensor.unsafe_reinitialize ();
      let row = "..d.. => ..d.." in
      let c23 () = NTDSL.init ~l:"c23" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 3 ] ~f:(fun _ -> 2.) () in
      let name = "rows, solved first" in
      let d = Shape.get_variable_ref "d" in
      let _y1 = capture ~spec:row d (c23 ()) in
      p
        (name ^ ": the first registration solved the reference")
        (Option.is_some d.var_ref.solved_dim);
      let w2 = open_leaf "w2" in
      let _y2 = capture ~spec:row d w2 in
      show name [ ("w2", w2) ];
      p (name ^ ": the open operand takes the solved row") (List.equal Int.equal (dims w2) [ 2; 3 ]);
      let name = "rows, after a closure" in
      let d = Shape.get_variable_ref "d" in
      let y1 = capture ~spec:row d (c23 ()) in
      ignore (dims y1 : int list);
      let w2 = open_leaf "w2" in
      let _y2 = capture ~spec:row d w2 in
      show name [ ("w2", w2) ];
      p (name ^ ": the later operand takes the closed product") (product w2 = 6));
  Verdict.case "shared reference, rows holding a symbolic extent" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "rows holding a symbolic extent" in
      let module Idx = Ir.Indexing in
      let seq, _bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
      let e = Shape.get_variable_ref "e" in
      Shape.set_sym_dim e seq;
      let w = open_leaf "w" in
      let _ = capture ~spec:"e=>e" e w in
      let d = Shape.get_variable_ref "d" in
      let _y1 = capture ~spec:"..d.. => ..d.." d w and _y2 = capture ~spec:"..d.. => ..d.." d w in
      show name [ ("w", w) ];
      p
        (name ^ ": two captures of one row close without a shape error")
        (List.equal Int.equal (dims w) [ 6 ]));
  Verdict.case "shared reference, executed" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "executed" in
      let d = Shape.get_variable_ref "d" in
      let w1 = open_leaf "w1" and w2 = open_leaf "w2" in
      let y1 = capture d w1 and y2 = capture d w2 in
      unsolved name d;
      fix w2 [ 5 ];
      observe y1;
      observe y2;
      let ctx = Train.forward_once (Context.auto ()) y1 in
      let ctx = Train.forward_once ctx y2 in
      List.iter
        [ ("y1", y1); ("y2", y2) ]
        ~f:(fun (label, y) ->
          p_all ~min:5
            (Printf.sprintf "%s: %s computes all 5 cells of the shared width" name label)
            (Array.to_list (Context.get_values ctx y.Tensor.value))
            ~f:(Float.equal 2.));
      Context.release ctx)
