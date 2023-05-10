(** Synchronizers are like semaphores, but instead of guarding N resources, they guard
    a synchronization point among N threads. *)

type t = {
  mut : Mutex.t; (* protects [v] *)
  mutable v : int; (* the current value *)
  zero : Condition.t; (* signaled when [v = 0] *)
}

let make v =
  if v < 1 then invalid_arg "Synchronizer.make: wrong initial value";
  { mut = Mutex.create (); v; zero = Condition.create () }

let synchronize s =
  Mutex.lock s.mut;
  if s.v = 0 then (
    Mutex.unlock s.mut;
    invalid_arg "Synchronizer.acquire: synchronization point already met.");
  s.v <- s.v - 1;
  if s.v = 0 then Condition.signal s.zero;
  while s.v > 0 do
    Condition.wait s.zero s.mut
  done;
  s.v <- s.v + 1;
  Mutex.unlock s.mut

let get_value s = s.v
