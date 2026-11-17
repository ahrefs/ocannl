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

   What changed with gh-ocannl-2247: a draw read through an einsum spec ([u ++ "ab => ba"]) is
   solved to the spec's axes, minted for the operation's result, and they closed minimal: the draw
   was one value under [c[|5,3]]. A solved row now passes its mark to the axes it absorbs, so they
   resolve at their use sites as the draw does.

   What changed with gh-ocannl-2229: a fixed-index input's minimum above 1 closes before
   unconstrained results guess minimal. The concrete size then travels through the existing
   leaf-before-result stages, so both two-result and three-result chains fill the indexed axis. *)

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
  draws "(u *. 2) *. 2, then a fixed index, batch 4" ~want:"4|4" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 (times2 u) in
      let%op y = m ++ "...|3 => ...|0" in
      ([ u ], m, y));
  draws "((u *. 2) *. 2) *. 2, then a fixed index, batch 4" ~want:"4|4" ~fills:(Some true)
    (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 (times2 (times2 u)) in
      let%op y = m ++ "...|3 => ...|0" in
      ([ u ], m, y));
  draws "u *. 2, then a fixed index, batch 4" ~want:"4|4" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 u in
      let%op y = m ++ "...|3 => ...|0" in
      ([ u ], m, y));
  (* gh-ocannl-2247: a draw read through an einsum spec is solved to the spec's axes; they take the
     draw's role and fill the use site. *)
  draws "(u ++ \"ab => ba\") *. c[|5,3]" ~want:"3,5" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = NTDSL.reshape ~l:"c" ~b:[] ~o:[ 5; 3 ] (values 15) () in
      let%op y = (u ++ "ab => ba") *. c in
      ([ u ], y, y));
  draws "(u ++ \"ab => a\") *. c[5|3]" ~want:"5|3,1" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let%op y = (u ++ "ab => a") *. c53 () in
      ([ u ], y, y))

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

(* gh-ocannl-2225: these formerly refused or repeated over an unbatched use site. Both leaf
   constructors and both creation orders already fill after gh-ocannl-2244/2247/2229. Build the
   whole noise expression before or after the leaf, so its intermediate rows change age too. Only
   the draws' values are claimed: nested log is undefined on these uniform draws. *)
let () =
  let programs =
    [
      ( "scaled log",
        fun () ->
          let u = TDSL.uniform () () in
          let%op n = 2. *. log u in
          ([ u ], n) );
      ( "box muller",
        fun () ->
          let u = TDSL.uniform () () and v = TDSL.uniform () () in
          let%op n = -2. *. log (1. + (2. *. u)) *. cos v in
          ([ u; v ], n) );
      ( "sum",
        fun () ->
          let u = TDSL.uniform () () and v = TDSL.uniform () () in
          let%op n = u + v in
          ([ u; v ], n) );
      ( "nested log",
        fun () ->
          let u = TDSL.uniform () () in
          let%op n = log (log u) in
          ([ u ], n) );
    ]
  in
  List.iter [ false; true ] ~f:(fun reshape ->
      List.iter [ false; true ] ~f:(fun leaf_first ->
          List.iter programs ~f:(fun (program, make_noise) ->
              let name =
                Printf.sprintf "%s, %s, %s" program
                  (if reshape then "reshape" else "term_init")
                  (if leaf_first then "leaf first" else "draws first")
              in
              draws name ~want:"16" ~fills:(Some true) (fun () ->
                  let make_leaf () =
                    let ones = Array.create ~len:16 1. in
                    if reshape then
                      NTDSL.reshape ~l:"x" ~b:[] ~o:[ 16 ]
                        (Ir.Ndarray.init_array ~debug:"x" Ir.Ops.single ~dims:[| 16 |] ~padding:None
                           ~f:(fun _ -> 1.))
                        ()
                    else Tensor.term_init ones ()
                  in
                  let used, noise, x =
                    if leaf_first then
                      let x = make_leaf () in
                      let used, noise = make_noise () in
                      (used, noise, x)
                    else
                      let used, noise = make_noise () in
                      let x = make_leaf () in
                      (used, noise, x)
                  in
                  let%op y = x *. noise in
                  (used, y, y)))))

(* gh-ocannl-2224: reducing the second consumer keeps its axis separate from the concrete bound.
   Without that reduction the final pointwise result sizes both consumers alike. *)
let () =
  draws "(u *. c[|3]) + reduce (u + 1), mixed separate bounds" ~want:"3" ~fills:(Some true)
    (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 3 ] in
      let%op r = plus1 u ++ "a => 0" in
      let%op y = (u *. c) + r in
      ([ u ], y, y));
  draws "reduce (u + 1) + (u *. c[|3]), mixed separate bounds reversed" ~want:"3" ~fills:(Some true)
    (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 3 ] in
      let%op r = plus1 u ++ "a => 0" in
      let%op y = r + (u *. c) in
      ([ u ], y, y))

(* The varying constant would make the sum vary even with a scalar draw. Check the materialized draw
   against the use site's cells. *)
let () =
  draws "relu u + (c[3] ++ \"a => a...\")" ~want:"3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 3 ] in
      let%op y = relu u + (c ++ "a => a...") in
      ([ u ], y, y));
  draws "(c[3] ++ \"a => a...\") + relu u" ~want:"3" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 3 ] in
      let%op y = c ++ "a => a..." + relu u in
      ([ u ], y, y));
  draws "relu u + (c[2,5] ++ \"ab => ab...\")" ~want:"2,5" ~fills:(Some true) (fun () ->
      let u = TDSL.uniform () () in
      let c = const [ 2; 5 ] in
      let%op y = relu u + (c ++ "ab => ab...") in
      ([ u ], y, y))

(* gh-ocannl-2224: the late-tail boundary is staged deliberately: ordinary program constraints
   register before closing. The GLB has a leading axis over a tail held open through Stage6a; supply
   the tail equation at Stage6b. Concrete axes already work through generic elimination, but an
   unmarked variable bounded by 3 waited until stage 7 and closed to 1. The packed role arm must
   retry after stage 6a. No natural program reaching this exact late-tail path was found; the staged
   solver claims pin it and the configured draws are execution smoke checks. *)
module Row = Ocannl.Row

let late_tail_dims ?(variable_axis = false) () =
  let prov = Row.provenance ~sh_id:2224 ~kind:`Output in
  let fresh () = Row.get_row_for_var prov (Row.get_row_var ()) in
  let draw = fresh () and tail = fresh () in
  let v = match draw.bcast with Row_var v -> v | Broadcastable -> assert false in
  Row.add_resolve_at_use v;
  let concrete = Row.get_default_dim ~d:3 () in
  let axis = if variable_axis then Row.Var (Row.get_var ()) else concrete in
  let glb = { tail with beg_dims = [ axis ] } in
  let fixture = if variable_axis then "late-tail variable fixture" else "late-tail fixture" in
  let counter = Row.get_var () in
  let origin = [] in
  let constraints =
    (if variable_axis then
       [
         Row.Dim_ineq { res = concrete; opnd = axis; from_ = Sexp.Atom "late-tail"; origin };
         Row.Shape_row ({ glb with bcast = Broadcastable }, origin);
       ]
     else [])
    @ [
        Row.Shape_row (draw, origin);
        Row.Row_ineq { res = glb; opnd = draw; origin };
        Row.Rows_constr
          {
            r = [ draw ];
            constr =
              Total_elems
                {
                  numerator =
                    Strided_var
                      {
                        coeff = Utils.safe_lazy "late-tail block" (fun () -> 4);
                        var = counter;
                        denom = 1;
                        round_up = true;
                      };
                  divided_by = [];
                  keep_axis = false;
                };
            origin;
          };
      ]
  in
  let pending, env =
    List.fold [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6a ]
      ~init:(constraints, Row.empty_env) ~f:(fun (cs, env) stage ->
        Row.solve_inequalities ~stage cs env)
  in
  p
    (fixture ^ ": the packed draw is open after Stage6a")
    (match (Row.subst_row env draw).bcast with Row_var _ -> true | Broadcastable -> false);
  let empty = { tail with bcast = Broadcastable } in
  let pending, env =
    Row.solve_inequalities ~stage:Stage6b
      (Row.Row_eq { r1 = tail; r2 = empty; origin } :: pending)
      env
  in
  let solved = Row.subst_row env draw in
  p
    (fixture ^ ": the closed GLB is retained at Stage6b")
    (Row.is_broadcastable solved.bcast
    && List.equal Int.equal
         (List.map (solved.beg_dims @ solved.dims) ~f:(function Row.Dim { d; _ } -> d | _ -> 0))
         [ 3 ]);
  p
    (fixture ^ ": three cells use one counter block")
    (Option.equal Int.equal (Row.get_dim_val env counter) (Some 1));
  let pending, env = Row.solve_inequalities ~stage:Stage7 pending env in
  Shape.refuse_unsolved ~where:"late packed draw tail" pending;
  let solved = Row.subst_row env draw in
  List.map (solved.beg_dims @ solved.dims) ~f:(fun d ->
      match d with Row.Dim { d; _ } -> d | _ -> failwith "late-tail dimension unresolved")

let () =
  draws "packed draw using late-tail fixture dimensions (smoke)" ~want:"3" ~fills:(Some true)
    (fun () ->
      let dims = late_tail_dims () in
      let u = TDSL.uniform () ~output_dims:dims () in
      let c = const [ 3 ] in
      let%op y = u *. c in
      ([ u ], y, y))

let () =
  draws "packed draw using bounded-variable late-tail fixture dimensions (smoke)" ~want:"3"
    ~fills:(Some true) (fun () ->
      let dims = late_tail_dims ~variable_axis:true () in
      let u = TDSL.uniform () ~output_dims:dims () in
      let c = const [ 3 ] in
      ([ u ], c, u))
