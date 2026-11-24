open Base
open Ocannl

type outcome = Refused_at_construction of string | Refused_later of string | Ran

let error_message = function
  | Row.Shape_error (msg, _) -> msg
  | exn -> Stdlib.Printexc.to_string exn

(* A solver refusal can leave the open suffix tainted (gh-ocannl-1301). The prior closure is still
   usable, but the suffix has to be discarded before the next case can build shapes. *)
let with_rewind f =
  Exn.protect ~f ~finally:(fun () ->
      if Option.is_some (Shape.taint_provenance ()) then Tensor.rewind_to_last_closure ())

(* [build] constructs an operation and returns its compile-and-run thunk. A caller expecting refusal
   at one phase can distinguish it from failure at the other. *)
let expect_shape_error build =
  with_rewind (fun () ->
      match build () with
      | exception Row.Shape_error (msg, _) -> Refused_at_construction msg
      | run -> ( match run () with () -> Ran | exception exn -> Refused_later (error_message exn)))
