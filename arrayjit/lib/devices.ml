open Base

module Gccjit_device : Backend.No_device_backend with type context = Exec_as_gccjit.context = struct
  type context = Exec_as_gccjit.context
  type compiled = Exec_as_gccjit.compiled = { context : context; run : unit -> unit }

  open Exec_as_gccjit

  let initialize () = ()
  let unsafe_cleanup () = ()
  let init = init
  let finalize = finalize

  let compile context ~name ?verbose code =
    jit context ~name ?verbose @@ Assignments.compile_proc ~name ?verbose code

  let from_host = from_host
  let to_host = to_host
  let merge = merge
end

module Gccjit_backend = Backend.Multicore_backend (Gccjit_device)
