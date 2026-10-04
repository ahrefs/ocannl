open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX

(* Test 1: RAW dependency — sgd_routine reads gradients written by grad_routine. Pattern from
   zero2hero_1of7_exec.ml simple_gradients_hosted. *)
let test_raw_dependency () =
  printf "=== Test 1: RAW dependency ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized l;
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_ctx, grad_routine = Train.to_routine ctx IDX.empty grad in
  let _, sgd_routine = Train.to_routine grad_ctx IDX.empty sgd in
  let grad_id = grad_routine.Context.routine_id in
  let sgd_deps = sgd_routine.Context.execution_deps in
  Verdict.p "sgd depends on grad" (Set.mem sgd_deps grad_id);
  Verdict.p "sgd has deps" (not (Set.is_empty sgd_deps));
  (* Correct order: grad then sgd *)
  let ctx' = Context.run ctx grad_routine in
  let _ctx' = Context.run ctx' sgd_routine in
  printf "Correct order (grad then sgd): OK\n"

(* Test 2: Disjoint routines — sibling compiles from same context produce no deps *)
let test_disjoint () =
  printf "\n=== Test 2: Disjoint routines ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op loss_x = { x_param = [ 5 ] } *. { x_in = [ 3 ] } in
  let%op loss_y = { y_param = [ 7 ] } *. { y_in = [ 2 ] } in
  Train.every_non_literal_materialized loss_x;
  Train.every_non_literal_materialized loss_y;
  let grad_x = Train.grad_update loss_x in
  let grad_y = Train.grad_update loss_y in
  let ctx = Train.init_params ctx IDX.empty loss_x in
  let ctx = Train.init_params ctx IDX.empty loss_y in
  (* Compile from same context — sibling branches, should be independent *)
  let _, routine_x = Train.to_routine ctx IDX.empty grad_x in
  let _, routine_y = Train.to_routine ctx IDX.empty grad_y in
  let x_id = routine_x.Context.routine_id in
  let y_id = routine_y.Context.routine_id in
  let x_deps = routine_x.Context.execution_deps in
  let y_deps = routine_y.Context.execution_deps in
  (* Neither should depend on the other — they may have deps on init_params routines *)
  Verdict.p "x does not depend on y" (not (Set.mem x_deps y_id));
  Verdict.p "y does not depend on x" (not (Set.mem y_deps x_id));
  (* Both should be runnable since init_params already executed *)
  Verdict.p "can_run x" (Context.can_run ctx routine_x);
  Verdict.p "can_run y" (Context.can_run ctx routine_y);
  (* Either order should work *)
  let ctx' = Context.run ctx routine_y in
  let _ctx' = Context.run ctx' routine_x in
  printf "Reverse order (y then x): OK\n"

(* Test 3: can_run reflects execution state *)
let test_can_run () =
  printf "\n=== Test 3: can_run ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_ctx, grad_routine = Train.to_routine ctx IDX.empty grad in
  let _, sgd_routine = Train.to_routine grad_ctx IDX.empty sgd in
  Verdict.p "can_run grad (before execution)" (Context.can_run ctx grad_routine);
  Verdict.p "cannot run sgd (before grad)" (not (Context.can_run ctx sgd_routine));
  let ctx' = Context.run ctx grad_routine in
  Verdict.p "can_run sgd (after grad)" (Context.can_run ctx' sgd_routine)

(* Test 4: Negative test — running a dependent routine out of order raises Failure *)
let test_wrong_order_raises () =
  printf "\n=== Test 4: Wrong order raises ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized l;
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_ctx, _grad_routine = Train.to_routine ctx IDX.empty grad in
  let _, sgd_routine = Train.to_routine grad_ctx IDX.empty sgd in
  (* sgd depends on grad — running sgd first must fail *)
  let is_enforcement =
    match Context.run ctx sgd_routine with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"Context.run:"
  in
  Verdict.p "Wrong order raises Failure from Context.run" is_enforcement

(* Test 5: Re-execution pattern — grad -> sgd -> grad succeeds without reset *)
let test_reexecution () =
  printf "\n=== Test 5: Re-execution (grad -> sgd -> grad) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_ctx, grad_routine = Train.to_routine ctx IDX.empty grad in
  let _, sgd_routine = Train.to_routine grad_ctx IDX.empty sgd in
  let ctx' = Context.run ctx grad_routine in
  let ctx' = Context.run ctx' sgd_routine in
  let _ctx' = Context.run ctx' grad_routine in
  printf "grad -> sgd -> grad: OK\n"

(* Test 6: rollback withdraws an optimistic execution marking (gh-ocannl-536). [Context.run] marks a
   routine executed before a later [sync] can report an asynchronous failure, so a contained
   launch/sync rejection has to undo it — otherwise the next candidate the autotuner compiles in the
   same lineage sees a dependency satisfied by a routine that never completed. *)
let test_rollback_execution () =
  printf "\n=== Test 6: rollback of an execution marking ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_ctx, grad_routine = Train.to_routine ctx IDX.empty grad in
  let _, sgd_routine = Train.to_routine grad_ctx IDX.empty sgd in
  let ctx' = Context.run ctx grad_routine in
  Verdict.p "can_run sgd (after grad)" (Context.can_run ctx' sgd_routine);
  Context.rollback_execution ctx' grad_routine.Context.routine_id;
  Verdict.p "cannot run sgd (after rollback)" (not (Context.can_run ctx' sgd_routine));
  (* The ledger is shared by reference across the lineage, so the rollback is visible from the
     context the routine was compiled from as well. *)
  let ctx' = Context.run ctx' grad_routine in
  Verdict.p "re-running grad restores it" (Context.can_run ctx' sgd_routine)

(* Test 7: poisoning condemns the lineage (gh-ocannl-536). A launch or sync failure that may have
   left device buffers partially written has no restore path, so continuing to time candidates on
   that lineage would score suspect data. Every entrypoint refuses, naming the routine. *)
let test_poisoned_lineage () =
  printf "\n=== Test 7: poisoned lineage ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op l = e + { c = [ 10 ] } in
  Train.every_non_literal_materialized l;
  let grad = Train.grad_update l in
  let ctx = Train.init_params ctx IDX.empty l in
  let _, routine = Train.to_routine ctx IDX.empty grad in
  let ctx = Context.run ctx routine in
  Context.poison_lineage ctx ~routine_name:routine.Context.name (Failure "synthetic device failure");
  let refuses what f =
    match f () with
    | _ -> Printf.ksprintf Verdict.fail "%s: not refused on a poisoned lineage" what
    | exception Failure msg ->
        (* The row reports two properties of one refusal, so it keeps its shape and the claims sit
           beside it, on the same [let]-bound booleans the row prints. *)
        let names_routine = String.is_substring msg ~substring:routine.Context.name in
        let names_cause = String.is_substring msg ~substring:"synthetic device failure" in
        printf "%s refused, names the routine: %b, names the cause: %b\n" what names_routine
          names_cause;
        Verdict.claimf "%s refusal names the routine" what names_routine;
        Verdict.claimf "%s refusal names the cause" what names_cause
  in
  refuses "run" (fun () -> ignore (Context.run ctx routine));
  refuses "sync" (fun () -> Context.sync ctx);
  refuses "get_values" (fun () -> ignore (Context.get_values ctx l.Tensor.value));
  (* Same-backend [copy] dispatches through the backend's transfer machinery rather than [to_host],
     so guarding the host round-trip alone would let a poisoned source export a suspect buffer into
     a clean lineage. *)
  let clean = Context.auto () in
  refuses "copy out" (fun () -> ignore (Context.copy ~src:ctx ~dst:clean l.Tensor.value));
  refuses "copy in" (fun () -> ignore (Context.copy ~src:clean ~dst:ctx l.Tensor.value))

(* Test 8: a merge-buffer input is a real read edge on the transfer that filled the transient slab
   (gh-ocannl-766), not on a writer of the destination's ordinary tensor buffer. The source and
   destination have separate ledgers, and the consumer writes a third node, so an ordinary
   input/output hazard cannot accidentally supply either ordering. *)
let test_merge_buffer_read_dependency () =
  printf "\n=== Test 8: merge-buffer read dependency ===\n";
  Tensor.unsafe_reinitialize ();
  let%op merge_value = [ 0.; 0. ] + [ 0.; 0. ] in
  let%op source_values = [ 1.; 2. ] + [ 10.; 20. ] in
  let%op merge_output = [ 0.; 0. ] + [ 0.; 0. ] in
  let%op destination_tick = [ 40.; 2. ] + [ 1.; 0. ] in
  Train.set_materialized merge_value.Tensor.value;
  Train.set_materialized merge_output.Tensor.value;
  Train.set_materialized destination_tick.Tensor.value;
  let source_write = [%cd merge_value =: source_values] in
  let named_source_write name =
    { source_write with asgns = Ir.Assignments.Block_comment (name, source_write.asgns) }
  in
  let source_root = Context.auto () in
  let source_ctx, source_writer =
    Context.compile source_root (named_source_write "merge_source_writer_1") IDX.empty
  in
  let source_ctx, _future_source_writer =
    Context.compile source_ctx (named_source_write "merge_source_writer_2") IDX.empty
  in
  let sibling_source_ctx, _sibling_source_writer =
    Context.compile source_root (named_source_write "merge_source_writer_sibling") IDX.empty
  in
  let destination_ctx, destination_tick_writer =
    Train.to_routine (Context.auto ()) IDX.empty (Train.forward destination_tick)
  in
  let destination_ctx = Context.run destination_ctx destination_tick_writer in
  (try
     ignore
       (Context.copy ~into_merge_buffer:Copy ~src:source_ctx ~dst:destination_ctx
          merge_value.Tensor.value);
     Verdict.fail "merge transfer ran before its source writer"
   with Failure msg ->
     Verdict.p "merge transfer refuses an unexecuted source writer"
       (String.is_substring msg ~substring:"before source writer"));
  (* The compile frontier names the later, unexecuted writer, but re-running the earlier writer is
     what actually refreshes the source buffer. Merge readiness follows that execution state. *)
  ignore (Context.run source_ctx source_writer : Context.t);
  (* The sibling compile owns a different buffer for the same tnode. The shared ledger's executed
     writer from the first branch must not make this untouched sibling source look ready. *)
  (try
     ignore
       (Context.copy ~into_merge_buffer:Copy ~src:sibling_source_ctx ~dst:destination_ctx
          merge_value.Tensor.value);
     Verdict.fail "merge transfer used an executed writer from a sibling buffer"
   with Failure msg ->
     Verdict.p "merge transfer scopes runtime writers to the source buffer"
       (String.is_substring msg ~substring:"before source writer"));
  let merge_ctx =
    Context.copy ~into_merge_buffer:Copy ~src:source_ctx ~dst:destination_ctx
      merge_value.Tensor.value
  in
  let consumer_comp = [%cd merge_output =: merge_value.merge] in
  let consumer_comp =
    {
      consumer_comp with
      asgns = Ir.Assignments.Block_comment ("merge_dep_consumer", consumer_comp.asgns);
    }
  in
  let consumer_ctx, consumer = Context.compile merge_ctx consumer_comp IDX.empty in
  Verdict.p "merge consumer has exactly the transfer dependency"
    (Set.length consumer.Context.execution_deps = 1
    && not (Set.mem consumer.Context.execution_deps destination_tick_writer.Context.routine_id));
  Verdict.p "merge consumer can run after the transfer" (Context.can_run consumer_ctx consumer);
  (* An explicit write supersedes both compiled source writers. It can feed the next transfer, but
     not until the current transfer generation's consumer executes. *)
  let source_ctx = Context.set_values source_ctx merge_value.Tensor.value [| 5.; 6. |] in
  (try
     ignore
       (Context.copy ~into_merge_buffer:Copy ~src:source_ctx ~dst:consumer_ctx
          merge_value.Tensor.value);
     Verdict.fail "merge transfer overwrote a generation with a pending consumer"
   with Failure msg ->
     Verdict.p "merge overwrite refuses a pending consumer"
       (String.is_substring msg ~substring:"pending consumers"));
  let consumer_ctx = Context.run consumer_ctx consumer in
  Verdict.p "first merge consumer reads the executed writer's values"
    (Array.equal Float.equal
       (Context.get_values consumer_ctx merge_output.Tensor.value)
       [| 11.; 22. |]);
  (* A write-free failure rollback removes the optimistic execution claim and its runtime-writer
     mapping. The explicit value that preceded the failed run remains a valid transfer source. *)
  ignore (Context.run source_ctx source_writer : Context.t);
  Context.rollback_execution source_ctx source_writer.Context.routine_id;
  let rollback_merge_ctx =
    Context.copy ~into_merge_buffer:Copy ~src:source_ctx ~dst:consumer_ctx merge_value.Tensor.value
  in
  Verdict.p "overwriting invalidates the old merge consumer"
    (not (Context.can_run consumer_ctx consumer));
  let abandoned_ctx, abandoned = Context.compile rollback_merge_ctx consumer_comp IDX.empty in
  Verdict.p "rolled-back writer does not block the prior source value"
    (Context.can_run abandoned_ctx abandoned);
  (* Releasing an undispatched candidate unregisters its pending merge read. It must not strand the
     lineage's single-tenant slab forever. *)
  Context.release abandoned_ctx;
  let source_ctx = Context.set_values source_ctx merge_value.Tensor.value [| 5.; 6. |] in
  let replacement_merge_ctx =
    Context.copy ~into_merge_buffer:Copy ~src:source_ctx ~dst:rollback_merge_ctx
      merge_value.Tensor.value
  in
  let replacement_ctx, replacement =
    Context.compile replacement_merge_ctx consumer_comp IDX.empty
  in
  let replacement_ctx = Context.run replacement_ctx replacement in
  Verdict.p "replacement merge consumer reads the explicit upload"
    (Array.equal Float.equal
       (Context.get_values replacement_ctx merge_output.Tensor.value)
       [| 5.; 6. |])

let () =
  Verdict.case "test_raw_dependency" test_raw_dependency;
  Verdict.case "test_disjoint" test_disjoint;
  Verdict.case "test_can_run" test_can_run;
  Verdict.case "test_wrong_order_raises" test_wrong_order_raises;
  Verdict.case "test_reexecution" test_reexecution;
  Verdict.case "test_rollback_execution" test_rollback_execution;
  Verdict.case "test_poisoned_lineage" test_poisoned_lineage;
  Verdict.case "test_merge_buffer_read_dependency" test_merge_buffer_read_dependency
