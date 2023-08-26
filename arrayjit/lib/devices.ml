open Base

module type Device = sig
  type t
  val jit : t -> name:string ->
    ?verbose:bool ->
    High_level.t -> unit -> unit
  val from_host : t -> Lazy_array.t -> unit
  val to_host : ?accum:Low_level.binop -> t -> Lazy_array.t -> unit
  val num_devices : unit -> int
  val get_id : t -> int
  val get_device : id:int -> t
end

