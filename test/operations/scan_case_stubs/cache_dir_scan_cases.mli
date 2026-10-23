(** The stand-in names the [{ocaml|...|ocaml}] snippets of [cache_dir_scan_cases.ml] call, for
    [tools/typecheck-scan-cases.sh], which compiles this interface first and opens it at the top of
    every snippet -- so an arm may rebind any of these names ([module Autotune = Other]), as it may
    over the real library. *)

(** What a cache-writing module offers: the direct store the scan's [~dir] arms call. *)
module type Store = sig
  val store : dir:string -> 'k -> 'v -> unit
end

module type S = Store
(** The signature the arms constrain by without declaring it ([module F (Schedule_cache : S)],
    [(module Schedule_cache : S)]); an arm that needs an unreadable [S] of another shape declares
    its own. *)

module Ir : sig
  module Schedule_cache : Store
end

module Cache : Store
(** A [Cache] in scope that is not the schedule cache: what the bare [Cache.store] after a local
    alias's scope ends ("a nested alias dies with its scope") calls in a program a compiler builds.
*)

module Autotune : sig
  val tune : cache_dir:string -> 'f -> unit
  val resolve_cache_dir : ?cache_dir:string -> search:bool -> unit -> string
end

(** The unrelated module the shadowing arms bind in the cache's or the tuner's place, so it offers
    both modules' operations, a value of the parameter's name, and a [Cache] of its own (what
    [M.Cache] names once [let module M = Other] rebinds [M]). *)
module Other : sig
  include Store
  module Cache : Store

  val tune : cache_dir:string -> 'f -> unit
  val resolve_cache_dir : ?cache_dir:string -> search:bool -> unit -> string
  val cache_dir : string
end

val get : arg_name:string -> default:string -> string
(** The configuration getter the built-in-default arms read. *)

val store : dir:string -> 'k -> 'v -> unit
(** A bare [store], called through no module. *)

val f : unit
(** The arguments the arms pass around. *)

val k : unit
val v : unit
val key : unit
val value : unit
val prefix : string
val suffix : string
val elsewhere : string
val compute : unit -> string
