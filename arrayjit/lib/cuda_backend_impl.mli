module Impl : Ir.Backend_impl.Lowered_backend

val host_capabilities : unit -> (string * string) list
(** The host-side capability probes of this backend, read without opening a device (gh-ocannl-1038;
    {!Context.host_capabilities} collects them). Empty where the backend's library is not linked. *)
