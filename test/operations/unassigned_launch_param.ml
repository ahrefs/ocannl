(* gh-ocannl-940: a launch parameter the caller never assigned is refused, naming it, rather than
   launched at 0.

   Backend linking used to initialize every bound symbol's cell to 0, so a caller who compiled with
   two static indices and forgot to write one got a launch at index 0 -- the first slice, the empty
   extent -- that no check could tell from a caller who meant 0: [Context.check_launch_bindings]
   validates the VALUE, and 0 passes every value check. Linking now leaves each cell unassigned
   ([Indexing.fresh_launch_param]), and the bind-time validation refuses an unassigned one.

   Pinned here, through the public path ([Context.compile], [Context.run]) on the configured
   backend. Forgetting both of two parameters is refused, naming both in one message. Forgetting one
   is refused before dispatch, naming the forgotten parameter and not the assigned one; the claim
   that nothing ran reads the accumulators afterwards, since a refusal raised after the kernel
   launched would have added to them. Writing 0 explicitly is a value like any other: the same
   routine then runs, and what it accumulated shows it ran at exactly the values written, 0
   included.

   The probe accumulates each bound value into its own scalar, so the executed claims are exact
   small integers. *)

open Base
open Stdio
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
open Verdict.Claims

let refusal f =
  match f () with
  | (_ : Context.t) -> None
  | exception Utils.User_error msg ->
      eprintf "refused: %s (not part of the golden)\n%!" msg;
      Some msg

let names msg (s : Ir.Indexing.static_symbol) =
  String.is_substring msg ~substring:(Ir.Indexing.symbol_ident s.Ir.Indexing.static_symbol ^ " ")

let () =
  let i, bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
  let j, bindings = IDX.get_static_symbol ~static_range:3 bindings in
  let sum_i = Train.loss_accumulator ~label:"sum_of_i" () in
  let sum_j = Train.loss_accumulator ~label:"sum_of_j" () in
  let%cd probe =
    ~~("unassigned launch param";
       sum_i =+ !@i;
       sum_j =+ !@j)
  in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx probe bindings in
  let ctx = Context.set_values ctx sum_i.Tensor.value [| 0. |] in
  let ctx = Context.set_values ctx sum_j.Tensor.value [| 0. |] in
  let read ctx tn = (Context.get_values ctx tn).(0) in
  let i_ref = IDX.find_exn routine.Context.bindings i in
  let j_ref = IDX.find_exn routine.Context.bindings j in
  (* Both forgotten: one refusal naming both. *)
  let both = refusal (fun () -> Context.run ctx routine) in
  p "a run with both parameters unassigned is refused" (Option.is_some both);
  p "that refusal names both parameters"
    (Option.exists both ~f:(fun msg -> names msg i && names msg j));
  (* One forgotten: [i] assigned, [j] not. *)
  i_ref := 2;
  let one = refusal (fun () -> Context.run ctx routine) in
  p "a run with one parameter unassigned is refused" (Option.is_some one);
  p "that refusal names the unassigned parameter" (Option.exists one ~f:(fun msg -> names msg j));
  p "that refusal does not name the assigned parameter"
    (Option.exists one ~f:(fun msg -> not (names msg i)));
  Context.sync ctx;
  p "the refused runs dispatched nothing"
    Float.(read ctx sum_i.Tensor.value = 0. && read ctx sum_j.Tensor.value = 0.);
  (* Zero, written explicitly, runs. *)
  j_ref := 0;
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let got_i = read ctx sum_i.Tensor.value and got_j = read ctx sum_j.Tensor.value in
  eprintf "backend=%s, sum_i=%.1f, sum_j=%.1f (not part of the golden)\n%!"
    (Context.backend_name ctx) got_i got_j;
  p "with 0 written explicitly the routine runs, at the assigned index" Float.(got_i = 2.);
  p "with 0 written explicitly the routine runs, at the explicit 0" Float.(got_j = 0.);
  (* And a written value persists to the next run: no re-assignment needed. *)
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  p "a written value persists across runs" Float.(read ctx sum_i.Tensor.value = 4.)
