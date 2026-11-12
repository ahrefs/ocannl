open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row

(* gh-ocannl-2154: a self-contraction followed by a pointwise use, satisfiable but once refused. The
   first case adds a concrete witness to the ORIGINAL registered store with [infer_equal]: these are
   additional equations, not a replacement of the operations or their constraints; with ordinary
   [1_default] singletons in place of the broadcast top the witness is refused. The refusal came
   from the stage-6 trial of the einsum's balanced mixed-flank residual [[w].<c> = <m>.[a]]: it
   committed the empty overlap, and the queued closing of p's middle row to its use-site bound
   [<r>.[4]] (that row shares [st]'s resolve-at-use mark) then refuted it.
   [Row.solve_closing_stages] replays the closing with that trial one overlap larger; the second
   case executes what inference now infers, against a host oracle computed from the inferred leaf's
   own shape and values. *)
let build () =
  let st = TDSL.uniform () () in
  let x = TDSL.range_of_shape ~output_dims:[ 3; 4; 5 ] () in
  let%op p = st *. !.2. in
  let%op y = p +* "...|h,w,..c..; ...|..c.. => ...|h" st in
  let%op q = p *. x in
  let%op z = y + q in
  (st, p, y, q, z)

let dims t = Array.to_list (Shape.to_dims t.Tensor.shape)
let show l = "[" ^ String.concat ~sep:";" (List.map l ~f:Int.to_string) ^ "]"

let witness ~one_basis =
  Tensor.unsafe_reinitialize ();
  try
    let st, pt, y, q, z = build () in
    let scalar = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[] () in
    let wide =
      TDSL.range_of_shape ~batch_dims:[] ~input_dims:[]
        ~output_axes:[ (one_basis, 1); (Row.default_basis, 5) ]
        ()
    in
    let one =
      TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_axes:[ (one_basis, 1) ] ()
    in
    let full = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3; 4; 5 ] () in
    Shape.infer_equal st.Tensor.shape scalar.Tensor.shape;
    Shape.infer_equal pt.Tensor.shape wide.Tensor.shape;
    Shape.infer_equal y.Tensor.shape one.Tensor.shape;
    Shape.infer_equal q.Tensor.shape full.Tensor.shape;
    Shape.infer_equal z.Tensor.shape full.Tensor.shape;
    ignore (Shape.to_dims z.Tensor.shape : int array);
    Ok
      (List.map
         [ (st, []); (pt, [ 1; 5 ]); (y, [ 1 ]); (q, [ 3; 4; 5 ]); (z, [ 3; 4; 5 ]) ]
         ~f:(fun (t, want) -> (dims t, want)))
  with Row.Shape_error (msg, trace) -> Error (Shape.error_to_string_hum msg trace)

let () =
  Verdict.case "concrete witness" @@ fun () ->
  let accepted =
    match witness ~one_basis:Row.bcast_if_1 with
    | Ok shapes -> shapes
    | Error msg ->
        Stdio.eprintf "witness refused: %s\n%!" msg;
        []
  in
  p_all ~min:5 "the original store accepts all five witness shapes" accepted ~f:(fun (got, want) ->
      List.equal Int.equal got want);
  p "control: the witness with ordinary 1_default singletons is refused"
    (Result.is_error (witness ~one_basis:Row.default_basis))

(* Right-aligned broadcasting of a row-major tensor of [dims] read at an index of [full]. *)
let broadcast_offset dims full idx =
  let skip = List.length full - List.length dims in
  List.foldi dims ~init:0 ~f:(fun i acc d ->
      let j = List.nth_exn idx (skip + i) in
      (acc * d) + if d = 1 then 0 else j)

let () =
  Verdict.case "inferred and executed" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  Row.reset_trial_replays ();
  let st, pt, y, q, z = build () in
  Train.set_materialized st.Tensor.value;
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  match Train.forward_once ctx z with
  | exception Row.Shape_error (msg, trace) ->
      Stdio.eprintf "refused: %s\n%!" (Shape.error_to_string_hum msg trace);
      p "the inferred program is accepted" false
  | ctx ->
      p "the inferred program is accepted" true;
      Stdio.printf "inferred shapes: st=%s p=%s y=%s q=%s z=%s\n"
        (show (dims st))
        (show (dims pt))
        (show (dims y))
        (show (dims q))
        (show (dims z));
      p "the closing replayed its balanced-flank choice" (Row.trial_replays () > 0);
      let sd = dims st and pd = dims pt and zd = dims z in
      p "p is the contraction's [h; w] over st's row"
        (List.length pd = List.length sd + 2 && List.equal Int.equal (List.drop pd 2) sd);
      p "z is x's [3; 4; 5]" (List.equal Int.equal zd [ 3; 4; 5 ]);
      let s = Context.get_values ctx st.Tensor.value
      and got = Context.get_values ctx z.Tensor.value in
      (* A zero draw would make the oracle's st terms vanish, whatever z held there. *)
      p_all ~min:1 "the leaf's draws are nonzero, so the oracle discriminates" (Array.to_list s)
        ~f:(fun v -> not (Float.equal v 0.));
      (* y[h] sums p[h, w, c] * st[c] = 2 * st[c]^2 over w and c, the same for every h. *)
      let w = List.nth_exn pd 1 in
      let y0 = Float.of_int w *. 2. *. Array.fold s ~init:0. ~f:(fun acc v -> acc +. (v *. v)) in
      (* x is [range_of_shape]: the cell's row-major offset, in z's order. *)
      let want =
        List.concat_map (List.range 0 3) ~f:(fun i ->
            List.concat_map (List.range 0 4) ~f:(fun j ->
                List.map (List.range 0 5) ~f:(fun k ->
                    let x = Float.of_int ((((i * 4) + j) * 5) + k) in
                    y0 +. (2. *. s.(broadcast_offset sd zd [ i; j; k ]) *. x))))
        |> Array.of_list
      in
      Stdio.eprintf "(not part of the golden) z[0..4]: %s\n%!"
        (String.concat ~sep:" " (List.init 5 ~f:(fun i -> Float.to_string got.(i))));
      p_close "z against the host oracle y + 2 * st * x" got want ~tol:(fun w ->
          1e-4 *. Float.max 1. (Float.abs w));
      ignore (q : Tensor.t)

(* A different rank mismatch stays refused, and costs no replay. *)
let () =
  Verdict.case "unrelated refusal" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  Row.reset_trial_replays ();
  let leaf = TDSL.range_of_shape ~output_dims:[ 3 ] () in
  let refused =
    try
      let%op bad = leaf ++ "...|a,b,..r.. => ...|a" in
      ignore (Shape.to_dims bad.Tensor.shape : int array);
      None
    with Row.Shape_error (msg, _) -> Some msg
  in
  p "control: an unrelated rank mismatch is still refused"
    (Option.exists refused ~f:(String.is_prefix ~prefix:"Number of axes mismatch"));
  (* No balanced-flank trial was reached: the refusal cost one run, not a search. *)
  p "control: a refusal with no balanced-flank trial is not replayed" (Row.trial_replays () = 0)
