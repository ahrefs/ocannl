module Missing : functor
  (_ : sig
     val name : string
   end)
  -> Ir.Backend_impl.Lowered_no_device_backend
