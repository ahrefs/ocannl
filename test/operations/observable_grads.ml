(* Gradients carry observation intent ([Tnode.set_observable]) instead of a [Never_virtual] request
   (docs/proposals/context-scoped-memory-modes.md, Never_virtual audit). This pins the per-lineage
   placement a parameter gradient gets in the archetypal flows:

   1. Fused fwd+bwd+sgd compiled as ONE routine: the gradient is written and consumed within the
   routine, so the lineage is free to virtualize it. (Under the retired global settlement,
   Never_virtual forced it [Local] -- the UNOBSERVABLE class -- despite gradients being the
   canonical thing users inspect; that was the "forcing a bad memory mode" spot.)

   2. Split flow compiled with raw [Context.compile] (no [Train.to_routine] output-guessing): the
   grad_update routine has no in-routine reader of the gradient, so this exposes what the resolver
   does for a cross-routine gradient without any materialization hint.

   3. Split flow via [Train.to_routine]: written-not-read nodes are guessed as outputs and get
   [set_materialized], so the gradient is [On_device] by declared intent and readable across
   routines and from the host. *)

open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let grad_of p = (Option.value_exn p.Tensor.diff).Tensor.grad

let param_by_label l name =
  List.find_exn (Set.to_list l.Tensor.params) ~f:(fun t ->
      String.equal (Ir.Tnode.label_name t.Tensor.value) name)

let show_placement label ctx tn =
  (* [in context] is descriptive -- a virtualized gradient legitimately has no context entry, which
     is the point of legs 1 and 2. Observation intent is the claim this test exists to pin, on every
     lineage, so it is asserted here while the line keeps its shape. *)
  let observable = Tn.is_observable tn in
  printf "%s placement: %s; in context: %b; observable intent: %b\n" label
    (Tn.Placements.debug (Context.placements ctx) tn)
    (Context.mem ctx tn) observable;
  Verdict.claimf "%s carries observation intent" label observable

let show_values label ctx tn =
  printf "%s:" label;
  Array.iter (Context.get_values ctx tn) ~f:(fun v -> printf " %.2f" v);
  printf "\n"

let fused () =
  printf "=== 1. Fused fwd+bwd+sgd (one routine) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op l = { p = [ 2.; 3. ] } *. { xin = [ 4.; 5. ] } in
  let update = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let ctx, routine = Train.to_routine ctx IDX.empty (Asgns.sequence [ update; sgd ]) in
  let ctx = Context.run ctx routine in
  let p = param_by_label l "p" in
  show_placement "fused grad" ctx (grad_of p);
  show_values "param after one fused step (expect p - 0.1*xin = 1.6 2.5)" ctx p.Tensor.value

let split_raw () =
  printf "=== 2. Split grad_update / sgd_update (raw Context.compile) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op l = { q = [ 2.; 3. ] } *. { yin = [ 4.; 5. ] } in
  let update = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let q = param_by_label l "q" in
  try
    let ctx = Train.init_params ctx IDX.empty l in
    let ctx, gu_routine = Context.compile ctx update IDX.empty in
    let ctx = Context.run ctx gu_routine in
    show_placement "split(raw) grad after grad_update" ctx (grad_of q);
    let ctx, sgd_routine = Context.compile ctx sgd IDX.empty in
    let ctx = Context.run ctx sgd_routine in
    show_placement "split(raw) grad after sgd" ctx (grad_of q);
    show_values "param after one split step (expect q - 0.1*yin = 1.6 2.5)" ctx q.Tensor.value
  with e -> printf "split(raw) raised: %s\n" (Exn.to_string e)

let split_to_routine () =
  printf "=== 3. Split via Train.to_routine (output-guessing materializes) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op l = { r = [ 2.; 3. ] } *. { zin = [ 4.; 5. ] } in
  let update = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let gu_ctx, gu_routine = Train.to_routine ctx IDX.empty update in
  let sgd_ctx, sgd_routine = Train.to_routine gu_ctx IDX.empty sgd in
  (* Run and probe through the routines' own contexts: those carry the allocations. *)
  let (_ : Context.t) = Context.run gu_ctx gu_routine in
  let ctx = Context.run sgd_ctx sgd_routine in
  let r = param_by_label l "r" in
  show_placement "split(to_routine) grad" ctx (grad_of r);
  show_values "grad values (= zin)" ctx (grad_of r);
  show_values "param after one split step (expect r - 0.1*zin = 1.6 2.5)" ctx r.Tensor.value

let () =
  fused ();
  split_raw ();
  split_to_routine ()
