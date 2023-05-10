open Base
(** Synchronizers are like semaphores, but instead of guarding N resources, they guard
    a synchronization point among N threads. *)

type t = { task_stages : int array }

let make v =
  if v < 1 then invalid_arg "Synchronizer.make: wrong initial value";
  { task_stages = Array.create ~len:v 0 }

let synchronize ~task_id ~stage s =
  if s.task_stages.(task_id) >= stage then
    invalid_arg "Synchronizer.synchronize: task already at or past the given stage";
  s.task_stages.(task_id) <- stage;
  while Array.exists s.task_stages ~f:(fun other_stage -> other_stage >= 0 && other_stage < stage) do
    Caml.Domain.cpu_relax ()
  done

let synchronize_and_reset ~task_id s =
  let final_stage = s.task_stages.(task_id) in
  if final_stage > 0 then (
    s.task_stages.(task_id) <- -1;
    while Array.exists s.task_stages ~f:(fun other_stage -> other_stage >= final_stage) do
      Caml.Domain.cpu_relax ()
    done;
    s.task_stages.(task_id) <- 0)
