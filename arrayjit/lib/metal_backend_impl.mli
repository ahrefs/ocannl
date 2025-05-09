module Fresh : functor
  (_ : sig
     val config : Ir.Backend_intf.config
   end)
  -> Ir.Backend_impl.Lowered_backend
