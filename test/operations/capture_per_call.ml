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
