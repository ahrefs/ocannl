open Base

module type Device = sig
  type t
  type ndarray
  type on_device_arrays = ndarray array Map.M(Lazy_array).t

  type compiled = {
    source : Assignments.t;  (** Keeps the hosted ndarrays alive. *)
    on_devices : on_device_arrays;
        (** The compiler decides which arrays to maintain on the devices (across runs). *)
    run : unit -> unit;
  }

  val compile : t -> name:string -> ?verbose:bool -> Assignments.t -> on_device_arrays -> compiled
  val wait_for_all : unit -> unit

  val from_host : t -> on_device_arrays -> Lazy_array.t -> unit
  (** Potentially asynchronous. *)

  val to_host : ?accum:Low_level.binop -> t -> on_device_arrays -> Lazy_array.t -> unit
  (** Potentially asynchronous. *)

  val num_devices : unit -> int
  val get_num : t -> int
  val get_device : num:int -> t
  val init_devices : unit -> unit
end

module CPU (* : Device *) = struct
  module Domain = Domain [@warning "-3"]

  let num_domains = Domain.recommended_domain_count () - 1
end
