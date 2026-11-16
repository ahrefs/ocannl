(* Array.equal is supplied by Base on the repository's OCaml 5.3 floor. *)
module Array : sig
  val get : 'a array -> int -> 'a
  val equal : ('a -> 'a -> bool) -> 'a array -> 'a array -> bool
end

module Verdict : sig
  val claim : string -> bool -> unit

  module Claims : sig
    val p : string -> bool -> unit
    val gated : when_:bool -> on:string -> string -> bool -> unit
    val p_all : string -> 'a list -> f:('a -> bool) -> unit
  end
end

module Context : sig
  val get_values : unit -> unit -> float array
end

val ctx : unit
val tensor : unit
