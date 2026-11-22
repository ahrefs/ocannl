(* Type context for the migration protocol snippet; models only the runtime API it calls. *)
type node

module Ll_test : sig
  val operand : b:int list -> o:int list -> i:int list -> (int array -> float) -> int array -> float

  val matmul_products :
    m:int -> n:int -> k:int -> (int array -> float) -> (int array -> float) -> float list list

  val reduction_cells :
    out:int array -> over:int array -> (int array -> int array -> float) -> float list list

  val exact_partials : sig_bits:int -> float list -> bool
  val execute : name:string -> unit -> seed:unit list -> read:unit list -> float array list
  val virtualize : node -> unit
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
val check_reference : float array -> unit

module Other : module type of Ll_test

val ignore_comparison : string -> 'a array -> 'b array -> f:('a -> 'b -> bool) -> unit

module Context : sig
  type t
  type routine
  type prelowered
  type reuse
  type observability = Stored | Recomputable

  val compile : ?prelowered:prelowered -> ?reuse:reuse -> t -> unit -> unit -> t * routine
  val run : t -> routine -> t
  val observability : t -> node -> observability
  val get_values : t -> node -> float array
  val release : t -> unit
  val auto : unit -> t
end

val prelowered : Context.prelowered
val reuse : Context.reuse
val opaque_compile : Context.t -> node -> Context.t * Context.routine

(* gh-ocannl-2334: the support module the cross-module snippets call, a same-typed stand-in for a
   rebound name, and the [Train] once-runs. *)
module Exact_sums : sig
  val with_zero : float array -> int array -> float
  val check_sums : string -> float list list -> float array -> unit
  val check2 : string -> float list list -> float array -> unit
  val check3 : string -> float list list -> float array -> unit
  val self_check : unit -> unit
  val get_stored_values : Context.t -> node -> float array
end

module Other_sums : module type of Exact_sums

module Tensor : sig
  type diff = { grad : node }
  type t = { value : node; diff : diff option }
end

module Option : sig
  val value_exn : 'a option -> 'a
end

val build : unit -> Tensor.t

module Train : sig
  val forward : Tensor.t -> unit
  val forward_once : Context.t -> Tensor.t -> Context.t
  val update_once : ?output_cd_file:bool -> Context.t -> Tensor.t -> Context.t
  val set_materialized : node -> unit
end

module Other_train : module type of Train

module Ocannl : sig
  module Train : module type of Train
end
