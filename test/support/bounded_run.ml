(* A call that may never return -- a solver looping over cyclic bindings, say -- run on its own
   domain under a deadline, so that a hang is a failed claim rather than a stalled test
   (gh-ocannl-1790, shared by gh-ocannl-1889). The looping domain cannot be cancelled, so a missed
   deadline reports the claim [<label>: returns] as failed and exits the process at once. *)

open Base
open Verdict.Claims

let default_deadline_seconds = 30.

let run ?(deadline_seconds = default_deadline_seconds) label f =
  let outcome = Atomic.make None in
  let worker =
    Domain.spawn (fun () ->
        Atomic.set outcome (Some (match f () with v -> Ok v | exception e -> Error e)))
  in
  let deadline = Unix.gettimeofday () +. deadline_seconds in
  let rec wait () =
    match Atomic.get outcome with
    | Some result ->
        Domain.join worker;
        result
    | None when Float.( > ) (Unix.gettimeofday ()) deadline ->
        Stdio.eprintf "%s: still running after %.0f s; stopping the run here\n%!" label
          deadline_seconds;
        pf "%s: returns" label false;
        Stdlib.exit 1
    | None ->
        Unix.sleepf 0.01;
        wait ()
  in
  match wait () with Ok v -> v | Error e -> raise e
