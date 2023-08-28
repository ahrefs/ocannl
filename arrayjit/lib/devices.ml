open Base

module Gccjit_device : Backend.No_device_backend with type context = Exec_as_gccjit.context = struct
  type context = Exec_as_gccjit.context
  type compiled = Exec_as_gccjit.compiled = { context : context; run : unit -> unit }

  open Exec_as_gccjit

  let initialize = initialize
  let init = init
  let finalize = finalize

  let compile context ~name ?verbose code =
    jit context ~name ?verbose @@ Assignments.compile_proc ~name ?verbose code

  let unsafe_cleanup = unsafe_cleanup

  let from_host context la =
    ignore (context, la);
    failwith "NOT IMPLEMENTED YET"

  (** Potentially asynchronous. *)
  let to_host context ?accum la =
    ignore (context, accum, la);
    failwith "NOT IMPLEMENTED YET"
end

module Gccjit_backend = Backend.Multicore_backend (Gccjit_device)
