(* Type context for the migration protocol snippet; models only the runtime API it calls. *)
module Ll_test : sig
  val operand : b:int list -> o:int list -> i:int list -> (int array -> float) -> int array -> float

  val matmul_products :
    m:int -> n:int -> k:int -> (int array -> float) -> (int array -> float) -> float list list

  val reduction_cells :
    out:int array -> over:int array -> (int array -> int array -> float) -> float list list

  val exact_partials : sig_bits:int -> float list -> bool
  val execute : name:string -> unit -> seed:unit list -> read:unit list -> float array list
end

module type Additive = sig
  type t

  val zero : t
  val ( + ) : t -> t -> t
end

module Float : sig
  include Additive with type t = float

  val equal : float -> float -> bool
end

module Fn : sig
  val id : 'a -> 'a
end

module List : sig
  val iter : 'a list -> f:('a -> unit) -> unit
  val hd_exn : 'a list -> 'a
  val map : 'a list -> f:('a -> 'b) -> 'b list
  val sum : (module Additive with type t = 'a) -> 'b list -> f:('b -> 'a) -> 'a
end

val program : unit
val p_all : string -> 'a list -> f:('a -> bool) -> unit
val p_all2 : string -> 'a array -> 'b array -> f:('a -> 'b -> bool) -> unit

module Other : module type of Ll_test

val ignore_comparison : string -> 'a array -> 'b array -> f:('a -> 'b -> bool) -> unit

module Context : sig
  val compile : unit -> unit * unit
  val run : unit -> unit -> unit
  val get_values : unit -> unit -> float array
  val release : unit -> unit
  val auto : unit -> unit
end

(* gh-ocannl-2334: the support module the cross-module snippets call, a same-typed stand-in for a
   rebound name, and the [Train] once-runs. *)
module Exact_sums : sig
  val with_zero : float array -> int array -> float
  val check_sums : string -> float list list -> float array -> unit
  val check2 : string -> float list list -> float array -> unit
  val check3 : string -> float list list -> float array -> unit
  val self_check : unit -> unit
end

module Other_sums : module type of Exact_sums

module Tensor : sig
  type diff = { grad : unit }
  type t = { value : unit; diff : diff option }
end

module Option : sig
  val value_exn : 'a option -> 'a
end

val build : unit -> Tensor.t

module Train : sig
  val forward_once : unit -> Tensor.t -> unit
  val update_once : ?output_cd_file:bool -> unit -> Tensor.t -> unit
  val set_materialized : unit -> unit
end

module Other_train : module type of Train

module Ocannl : sig
  module Train : module type of Train
end
