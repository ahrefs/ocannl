module Fresh (Config : sig
  val config : Ir.Backend_intf.config
end) =
struct
  let _ = ignore Config.config

  include Lowered_backend_missing
end
