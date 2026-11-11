open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

(* gh-ocannl-2154. [p] closes to [st] and the scalar factor: its output rank is max(rank(st), 1).
   The spec requires rank(p) = rank(st) + 2. Setting h = 1 changes no rank, so it cannot make that
   equation hold under close-down. *)
let () =
  Verdict.case "contraction followed by pointwise use" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let refusal =
    try
      let st = TDSL.uniform () () in
      let x = TDSL.range_of_shape ~output_dims:[ 3; 4; 5 ] () in
      let%op p = st *. !.2. in
      let%op y = p +* "...|h,w,..c..; ...|..c.. => ...|h" st in
      let%op z = y + (p *. x) in
      ignore (Shape.to_dims z.Tensor.shape : int array);
      None
    with Row.Shape_error (msg, _) -> Some msg
  in
  p "the self-contraction followed by pointwise use is refused"
    (Option.exists refusal ~f:(String.is_prefix ~prefix:"Number of axes mismatch"));
  p "the refusal reports the open row's minimum rank and the closed row's rank"
    (Option.exists refusal
       ~f:(String.is_substring ~substring:"at least 1 axes, but the other row is closed with 0"));
  p "the refusal explains operation-result closing"
    (Option.exists refusal
       ~f:(String.is_substring ~substring:"Operation results close to their arguments' shapes"));
  p "the refusal names explicit use-site widening"
    (Option.exists refusal ~f:(String.is_substring ~substring:"use stretch"))
