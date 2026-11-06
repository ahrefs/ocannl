open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Idx = Ir.Indexing
module Row = Ocannl.Row
module Operation = Ocannl.Operation

(* gh-ocannl-1961: the executed side of the row solver's closures of a packed draw. A draw sized by
   inference must equal its twin given the same sizes explicitly: a closure by guess left it smaller
   and broadcast it, so every value downstream was right for some smaller draw and nothing but the
   draw's own element count tells. Shape and count are claimed with distinctness, because a smaller
   draw is distinct too. The row-level pins, with the state each closure enters, are in
   test/einsum/test_row_closure_guess.ml.

   [uniform1]'s equality specs never had this; it is the control where the claim is about the packed
   draw, and the route a symbolic extent has to take. *)

let dims t = Array.to_list (Shape.to_dims t.Tensor.shape)
let total dims = List.fold dims ~init:1 ~f:( * )
let unique values = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list values))

(* A value any backend computes, and a golden that cannot say which one did. *)
let context () =
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  ctx

let draw_of (build : unit -> Tensor.t * Tensor.t) =
  Tensor.unsafe_reinitialize ();
  let u, y = build () in
  Train.set_materialized u.Tensor.value;
  let ctx = Train.forward_once (context ()) y in
  (dims u, Context.get_values ctx u.Tensor.value)

let same_draw name ~inferred ~explicit =
  Verdict.case name @@ fun () ->
  let dims_i, values_i = draw_of inferred and dims_e, _ = draw_of explicit in
  pf "%s: the same shape as the explicit twin" name (List.equal Int.equal dims_i dims_e);
  pf "%s: a value for every cell" name (Array.length values_i = total dims_e);
  pf "%s: all distinct" name (unique values_i = Array.length values_i)

let values n =
  Ir.Ndarray.init_array ~debug:"t" Ir.Ops.single ~dims:[| n |] ~padding:None ~f:(fun idx ->
      Float.of_int idx.(0))

(* Sized by [Num_elems] at stage 3, or stage 5 with the batch row open too. *)
let late ?b n = NTDSL.reshape ~l:"t" ?b (values n) ()
let sized n = NTDSL.reshape ~l:"t" ~b:[] ~o:[ n ] (values n) ()

let fixed_index m =
  let%op y = m ++ "...|3 => ...|0" in
  y

(* (a): the fixed index's axis closes at the final stage, and a batch of 4 or 8 used to take the
   stage-7 guess, a batch of 3 not. *)
let () =
  List.iter [ 3; 4; 8 ] ~f:(fun b ->
      let draw ?output_dims () = TDSL.uniform () ~batch_dims:[ b ] ?output_dims () in
      let chain u =
        let%op m = u *. !.2. in
        (u, fixed_index m)
      in
      same_draw
        (Printf.sprintf "final-stage bound, batch %d" b)
        ~inferred:(fun () -> chain (draw ()))
        ~explicit:(fun () -> chain (draw ~output_dims:[ 4 ] ())));
  let stretched u =
    let%op m = stretch u in
    (u, fixed_index m)
  in
  same_draw "final-stage bound through stretch"
    ~inferred:(fun () -> stretched (TDSL.uniform () ~batch_dims:[ 4 ] ()))
    ~explicit:(fun () -> stretched (TDSL.uniform () ~batch_dims:[ 4 ] ~output_dims:[ 4 ] ()))

(* (b): a draw's input row meets the reshape through a pointwise result, so its bound arrives after
   stage 3, and a closed batch makes the row the only one that matters. *)
let () =
  let through_pointwise ?b t =
    let u = TDSL.uniform () ?batch_dims:b () in
    let%op y = (u + !.1.) * t in
    (u, y)
  in
  List.iter
    [ (None, "no batch"); (Some [ 2 ], "batch 2"); (Some [ 4 ], "batch 4") ]
    ~f:(fun (b, name) ->
      same_draw ("sibling bounded late, " ^ name)
        ~inferred:(fun () -> through_pointwise ?b (late ~b:[] 12))
        ~explicit:(fun () -> through_pointwise ?b (sized 12)));
  let through_stretch ?b t =
    let u = TDSL.uniform () ?batch_dims:b () in
    let%op y = stretch u * t in
    (u, y)
  in
  same_draw "sibling bounded at stage 6, through stretch"
    ~inferred:(fun () -> through_stretch ~b:[ 4 ] (sized 12))
    ~explicit:(fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] ~input_dims:[ 12 ] () in
      let t = sized 12 in
      let%op y = stretch u * t in
      (u, y));
  same_draw "sibling bounded late, sized with the batch row open too"
    ~inferred:(fun () ->
      let t = late 12 in
      through_pointwise ~b:[ 2 ] t)
    ~explicit:(fun () ->
      through_pointwise ~b:[ 2 ] (NTDSL.reshape ~l:"t" ~b:[ 2 ] ~o:[ 6 ] (values 12) ()));
  same_draw "the same, the reshape built before the draw and after it"
    ~inferred:(fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 2 ] () in
      let t = late 12 in
      let%op y = (u + !.1.) * t in
      (u, y))
    ~explicit:(fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 2 ] () in
      let t = NTDSL.reshape ~l:"t" ~b:[ 2 ] ~o:[ 6 ] (values 12) () in
      let%op y = (u + !.1.) * t in
      (u, y));
  let matmul t =
    let u = TDSL.uniform () () in
    let%op y = u * t in
    (u, y)
  in
  (* Refused outright before: the sibling closing empty took the reshape's row with it. *)
  same_draw "a sibling shared with a reshape, its size arriving at stage 3"
    ~inferred:(fun () -> matmul (late ~b:[] 12))
    ~explicit:(fun () -> matmul (sized 12));
  same_draw "a sibling shared with a reshape, its size arriving at stage 5"
    ~inferred:(fun () -> matmul (late 12))
    ~explicit:(fun () -> matmul (sized 12))

(* The issue's own probe: both rows of a packed parameter bounded through pointwise chains. Rows in
   a parameter never were closeable; it stays correct. *)
let () =
  let chained init =
    Tensor.unsafe_reinitialize ();
    TDSL.default_param_init := init;
    let t = late 12 in
    let s = NTDSL.reshape ~l:"s" (values 5) () in
    let%op y = s *. ({ w } + 1) * t in
    let ctx = Train.forward_once (context ()) y in
    let w =
      Set.find_exn y.Tensor.params ~f:(fun p ->
          String.is_prefix (Ir.Tnode.label_name p.Tensor.value) ~prefix:"w")
    in
    TDSL.default_param_init := Operation.default_uniform_param_init;
    (dims w, Context.get_values ctx w.Tensor.value)
  in
  List.iter
    [
      ("packed", Operation.default_uniform_param_init);
      ("uniform1", Operation.default_uniform1_param_init);
    ]
    ~f:(fun (name, init) ->
      Verdict.case ("parameter, " ^ name) @@ fun () ->
      let w_dims, w_values = chained init in
      pf "parameter bounded through chains, %s: [5; 12]" name
        (List.equal Int.equal w_dims [ 5; 12 ]);
      pf "parameter bounded through chains, %s: all distinct" name
        (Array.length w_values = 60 && unique w_values = 60))

(* (c): a statistic [..c..] used by a tensor over [h, w, ..c..] through a pointwise result. The
   tail's axes are the statistic's; the flanks are the rest of the spatial row. *)
let () =
  Verdict.case "refuted GLB" @@ fun () ->
  let d, _ =
    draw_of (fun () ->
        let st = TDSL.uniform () () in
        let x = TDSL.range_of_shape ~label:[ "x" ] ~output_dims:[ 3; 4; 5 ] () in
        let%op y = (st *. x) +* "...|h,w,..c..; ...|..c.. => ...|h" st in
        (st, y))
  in
  p "a GLB the rank facts refute leaves the statistic its own axes" (List.equal Int.equal d [ 5 ])

(* Least commitment: what no bound speaks to stays as it was. *)
let () =
  Verdict.case "least commitment" @@ fun () ->
  let d, values =
    draw_of (fun () ->
        let u = TDSL.uniform () () in
        (u, u))
  in
  p "a bare draw is one block" (List.equal Int.equal d [ 4 ] && unique values = 4);
  let d, _ =
    draw_of (fun () ->
        let u = TDSL.uniform () () in
        let%op y = u + !.1. in
        (u, y))
  in
  p "a draw meeting a scalar stays one element" (List.equal Int.equal d [ 1 ])

(* A symbolic extent sizes no total (gh-ocannl-817), through a use site as well as directly. The
   closed batch is the case that used to pass silently: 4 values across a 6-extent axis. *)
let () =
  Verdict.case "symbolic use sites" @@ fun () ->
  let refused y =
    match Train.forward_once (context ()) y with
    | _ -> false
    | exception Row.Shape_error (msg, _) ->
        String.is_substring msg ~substring:"not supported with symbolic dimensions"
  in
  let extent () =
    Tensor.unsafe_reinitialize ();
    fst (Idx.get_static_symbol ~static_range:6 Idx.Empty)
  in
  let symbol = extent () in
  let u = TDSL.uniform () () in
  let%op y = (u *. !.2.) ++ "s=>s" [ "s" ] in
  Shape.set_sym_dim s symbol;
  p "a draw over a symbolic use site is refused, naming why" (refused y);
  let symbol = extent () in
  let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
  let%op y = (u *. !.2.) ++ "...|s=>...|s" [ "s" ] in
  Shape.set_sym_dim s symbol;
  p "so is a batched one, whose total the closed axis divides" (refused y);
  let symbol = extent () in
  let u = TDSL.uniform1 () () in
  let%op y = (u *. !.2.) ++ "s=>s" [ "s" ] in
  Shape.set_sym_dim s symbol;
  let ctx = Train.forward_once (context ()) y in
  p "uniform1 is the route that sizes one" (Array.length (Context.get_values ctx y.Tensor.value) = 6)
