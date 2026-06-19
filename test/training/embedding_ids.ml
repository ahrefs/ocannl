(* gh-ocannl-343: fast training smoke test for the token-ID embedding path.

   A tiny model embeds token IDs via a logical one-hot ([range vocab == ids], optimized into a
   guarded gather), projects to a scalar, and regresses to per-example targets. We run a handful of
   SGD steps and assert the loss decreases and reaches near zero -- i.e. gradients flow back through
   the embedding lookup into the embedding table. This is deliberately tiny (seconds) so it gives
   fast feedback; the convergence-quality integration test lives in the slow [mlp_names] run. *)

open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

let () =
  Utils.settings.fixed_state_for_init <- Some 11;
  Tensor.unsafe_reinitialize ();

  let vocab = 6 and embed = 4 and n = 4 in
  (* Token IDs (one per example) and a target scalar per example. *)
  let ids = TDSL.ndarray [| 1.; 3.; 5.; 0. |] ~label:[ "ids" ] ~batch_dims:[ n ] ~output_dims:[] () in
  let target =
    TDSL.ndarray [| 1.; 0.; 1.; 0. |] ~label:[ "target" ] ~batch_dims:[ n ] ~output_dims:[ 1 ] ()
  in

  (* Embedding lookup as a logical one-hot times the table, then a linear projection to a scalar. *)
  let%op embedded = { c; o = [ embed ] } * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let%op pred = ({ w } * embedded) + { b = 0.; o = [ 1 ] } in
  let%op diff = pred - target in
  let%op batch_loss = ((diff *. diff) ++ "...|... => 0") /. !..n in

  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.5 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty batch_loss in
  let sgd_step = Train.to_routine ctx IDX.empty (Asgns.sequence [ update; sgd ]) in
  let ctx = Context.context sgd_step in
  let open Operation.At in

  Train.run ctx sgd_step;
  let initial = (ctx, batch_loss).@[0] in
  for _ = 1 to 60 do
    Train.run ctx sgd_step
  done;
  let final = (ctx, batch_loss).@[0] in
  printf "initial loss finite: %b\n" (Float.is_finite initial);
  printf "loss decreased: %b\n" Float.(final < initial);
  printf "loss converged near zero: %b\n" Float.(final < 0.01)
