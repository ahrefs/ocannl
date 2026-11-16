(* gh-ocannl-2126: a row's lower bound travels the whole chain of bounds, pinned on executed
   programs. A tensor that resolves at its use sites (a draw, a parameter) takes the bound however
   many operation results stand between it and the use site, and however the chain branches; the
   results close after their arguments, to what those require. The row-level stores are
   test/einsum/test_row_glb_reach.ml.

   Each draw case prints the draw's shape and how many distinct values it holds against how many
   cells its use site has; each parameter case prints the parameters' shapes.

   What changed with gh-ocannl-2126: [(u + v) *. c] was refused ("Too many axes in an operand": the
   result closed empty while the draws under it took the bound), a draw with two consumers closed to
   one element, and a parameter behind a pointwise result of two parameters, or behind a pointwise
   result of two results, was refused as a hidden dimension.

   What changed with gh-ocannl-2180: a numeric literal has one size-1 output axis, so [u + 1] has an
   output axis of its own, and the draw, closing to its bound, took that very axis, which closes
   minimal as the result's does: the batch filled and the output repeated. The draw now takes axes
   of its own, bounded by the result's, so it fills like the draw beside a constant with no axes;
   with two consumers each one's axis stays related to the draw.

   What changed with gh-ocannl-2244: a draw behind a unary result, behind a pointwise result of the
   draw with itself, or under a compose, closed to one element (or lost its output axes) when the
   use site's axes were concrete: the draw waited at stage 6a and the result above it, closing at
   6b, closed it empty.

   What still repeats, and why: in [(u *. 2) *. 2, then a fixed index] the use site's size is a
   fixed index's lower bound on the outer result's axis, which is no GLB and does not travel down to
   the inner result; that one closes minimal, and the draw matches it. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train

let context () =
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  ctx

let values n =
  Ir.Ndarray.init_array ~debug:"c" Ir.Ops.single ~dims:[| n |] ~padding:None ~f:(fun idx ->
      Float.of_int idx.(0))

let unique v = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list v))
let cells t = Array.fold (Shape.to_dims t.Tensor.shape) ~init:1 ~f:( * )
let shape t = Shape.to_string_hum t.Tensor.shape

(* Each draw's shape and distinct values, and the use site's cells. [build] returns the draws, their
   use site, and the tensor to run. *)
let observe build =
  Tensor.unsafe_reinitialize ();
  let draws, use, root = build () in
  List.iter draws ~f:(fun u -> Train.set_materialized u.Tensor.value);
  let ctx = Train.forward_once (context ()) root in
  (List.map draws ~f:(fun u -> (shape u, unique (Context.get_values ctx u.Tensor.value))), cells use)

(* [want] is the shape every draw of the case closes to. A draw that fills holds a value per cell of
   its use site; one that repeats holds fewer. [~fills:None] makes no such claim: where the use site
   has one cell, filling it is [1 = 1] whatever the solver did. *)
let draws name ~want ~(fills : bool option) build =
  Verdict.case name @@ fun () ->
  let seen, total = observe build in
  Stdio.printf "%s\n" name;
  List.iter seen ~f:(fun (shape, n) ->
      Stdio.printf "  draw = %s, %d distinct values for %d cells\n" shape n total);
  p_all ~min:1
    (name ^ ": every draw closes to " ^ want)
    seen
    ~f:(fun (shape, _) -> String.equal shape want);
  match fills with
  | None -> ()
  | Some true ->
      p_all ~min:1 (name ^ ": every draw fills its use site") seen ~f:(fun (_, n) -> n = total)
  | Some false ->
      p_all ~min:1 (name ^ ": every draw repeats along its use site") seen ~f:(fun (_, n) ->
          n < total)

(* The parameters' shapes after the program ran. *)
let params name ~want build =
  Verdict.case name @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let ps, root = build () in
  let (_ : Context.t) = Train.forward_once (context ()) root in
  let seen = List.map ps ~f:shape in
  Stdio.printf "%s\n  %s\n" name (String.concat ~sep:", " seen);
  pf_all2 "%s: the parameters take the shapes their use sites give" name (Array.of_list seen)
    (Array.of_list want) ~f:String.equal

let plus1 r =
  let%op r2 = r + !.1. in
  r2

let times2 r =
  let%op r2 = r *. !.2. in
  r2

let stretched r =
  let%op s = stretch r in
  s

let c53 () = NTDSL.reshape ~l:"c" ~b:[ 5 ] ~o:[ 3 ] (values 15) ()

(* An unbatched constant with output axes [o]: the use site's axes are concrete by the time the draw
   under a result closes. *)
let const o = NTDSL.reshape ~l:"c" ~b:[] ~o (values (List.fold o ~init:1 ~f:( * ))) ()

let () =
  draws "u *. c[5|3]" ~want:"5|3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let%op y = u *. c53 () in
      ([ u ], y, y));
  draws "stretch (stretch (stretch u)) *. c[5|3]" ~want:"5|3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let%op y = stretched (stretched (stretched u)) *. c53 () in
      ([ u ], y, y));
  (* A chain that branches: neither draw is the result's only argument. *)
  draws "(u + v) *. c[5|3], two draws" ~want:"5|3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () and v = TDSL.uniform () () in
      let%op y = (u + v) *. c53 () in
      ([ u; v ], y, y));
  (* A constant with no axes gives the result no axis of its own. *)
  draws "(u + z) *. c[5|3], z a constant with no axes" ~want:"5|3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let z = NTDSL.reshape ~l:"z" ~b:[] ~o:[] (values 1) () in
      let%op y = (u + z) *. c53 () in
      ([ u ], y, y));
  (* A literal has one size-1 output axis of its own, which the draw does not adopt. *)
  draws "(u + 1) *. c[5|3]" ~want:"5|3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let%op y = plus1 u *. c53 () in
      ([ u ], y, y));
  draws "((u + 1) + (u *. 2)) *. c[5|3], one draw with two consumers" ~want:"5|3" ~fills:(Some true)
    (fun () ->
      let u = TDSL.uniform () () in
      let%op y = (plus1 u + times2 u) *. c53 () in
      ([ u ], y, y));
  (* Unbatched: the draw's bound is the result's open row with one axis, no batch to fill. *)
  draws "(u + 1) *. c[16]" ~want:"16" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = NTDSL.reshape ~l:"c" ~b:[] ~o:[ 16 ] (values 16) () in
      let%op y = plus1 u *. c in
      ([ u ], y, y));
  (* Control: nothing sizes the draw, so it closes minimal with the literal's axis (an axis of its
     own would have been sized by the packed draw's block, 4). *)
  draws "u + 1, nothing sizing u" ~want:"1" ~fills:None (fun () ->
      let u = TDSL.uniform () () in
      let y = plus1 u in
      ([ u ], y, y));
  (* gh-ocannl-1961's executed shape: the use site's bound arrives at the final stage, through a
     fixed index, and a chain of two operation results stands between it and the draw. *)
  draws "(u *. 2) *. 2, then a fixed index, batch 4" ~want:"4|1" ~fills:(Some false) (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 (times2 u) in
      let%op y = m ++ "...|3 => ...|0" in
      ([ u ], m, y));
  draws "u *. 2, then a fixed index, batch 4" ~want:"4|4" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 u in
      let%op y = m ++ "...|3 => ...|0" in
      ([ u ], m, y))

(* gh-ocannl-2244: a draw whose use site's axes are concrete, behind a result: a unary result, one
   draw as both operands of a pointwise result, a compose result. The use site's bound reached the
   draw, but the draw, waiting on the open tail of that bound, was still open when the result above
   it closed, and the result closing empty closed the draw empty with it. *)
let () =
  draws "relu u *. c[|2,5]" ~want:"2,5" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2; 5 ] in
      let%op y = relu u *. c in
      ([ u ], y, y));
  draws "c[|2,5] + relu u" ~want:"2,5" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2; 5 ] in
      let%op y = c + relu u in
      ([ u ], y, y));
  draws "relu (relu u) *. c[|2,5]" ~want:"2,5" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2; 5 ] in
      let%op y = relu (relu u) *. c in
      ([ u ], y, y));
  draws "(u *. u) + c[|2]" ~want:"2" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2 ] in
      let%op y = (u *. u) + c in
      ([ u ], y, y));
  draws "(u + u) + c[|2]" ~want:"2" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2 ] in
      let%op y = u + u + c in
      ([ u ], y, y));
  (* The compose's draw takes its input axis from the contraction and its output axes from the outer
     use site: the whole shape is claimed, and the draw fills it. *)
  draws "(c[|3,2] *. v) *. (u * c[|5]), the compose's draw" ~want:"5->3,2" ~fills:(Some true)
    (fun () ->
      let u = TDSL.uniform () () and v = TDSL.uniform () () in
      let c32 = const [ 3; 2 ] and c5 = const [ 5 ] in
      let%op y = c32 *. v *. (u * c5) in
      ([ u ], u, y))

let () =
  (* The bias-like [1;2] sizes both parameters' outputs through the pointwise result of the two. *)
  params "[1;2] + (({w} + {v}) * [3;4;5])" ~want:[ "3->2"; "3->2" ] (fun () ->
      let%op y = [ 1; 2 ] + (({ w = uniform1 () } + { v = uniform1 () }) * [ 3; 4; 5 ]) in
      ([ w; v ], y));
  (* Through two unary results and a pointwise result of two results. *)
  params "[1;2] + relu (relu (({w} * [3;4;5]) *. ({v} * [6;7])))" ~want:[ "3->2"; "2->2" ]
    (fun () ->
      let%op h = { w = uniform1 () } * [ 3; 4; 5 ] *. ({ v = uniform1 () } * [ 6; 7 ]) in
      let%op y = [ 1; 2 ] + relu (relu h) in
      ([ w; v ], y))
