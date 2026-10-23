(** The stand-in names the [{ocaml|...|ocaml}] snippets of [config_scan_lexing.ml] call, for
    [tools/typecheck-scan-cases.sh], which compiles this interface first and opens it at the top of
    every snippet -- so an arm may rebind any of these names, as it may over the real library. *)

val get : arg_name:string -> default:string -> string
(** The getter the arms spell their reads with, typed as [Utils.get_global_arg] is. *)

val get_style : ?arg_name:string -> unit -> string
val g : ?arg_name:string -> unit -> string

(** The library modules the settings and generated-artifact arms read through. *)
module Utils : sig
  type settings = { large_models : bool }

  val settings : settings
  val debug_log_from_routines : unit -> bool
  val get_global_arg : arg_name:string -> default:string -> string
end

module Low_level : sig
  type virtualize_settings = { max_visits : int }

  val virtualize_settings : virtualize_settings
end

module Test_utils : sig
  module Generated : sig
    val init : backend_name:string -> unit
    val read : string -> string
  end
end

(** The unrelated module a [Generated] alias is bound to by name only. *)
module Somewhere : sig
  module Else : module type of Test_utils.Generated
end

(** Another module, whose reach the process-lifetime arms stand in for. *)
module Other : sig
  val fetch : unit -> string
  val helper : unit -> string
end

module type S = sig end
(** The signature a packed module is constrained by, unread. *)

type record = { field : int }
(** The record a punned literal [{ field }] builds, and the value it puns. *)

val field : int

val name : string
(** The arguments and plumbing the arms pass around: a name bound outside the snippet, the backend a
    generated-artifact arm initializes, a helper the snippet calls before (or without) defining it,
    and the opaque functions a lazy is handed to. *)

val backend_name : string
val helper : unit -> string
val switch : string Lazy.t -> unit
val probe : compute:(unit -> string) -> string
val resolve : arg_name:string -> bootstrap:string -> string

(* The arms no compiler accepts, each checked to stay rejected.

   Two pin what the PARSER makes of a functor application in expression position -- [Pexp_field]
   over a constructor application, never an applied [Pexp_ident] -- which is the premise
   [Config_key_scan.flatten_longident] rests on; the language produces no such program, which is the
   point of both. *)
[@@@known_rejection
"a functor-application elsewhere does not crash the scan", "Unbound constructor Set.Make"]

[@@@known_rejection
"an applied path in expression position is not an identifier, so not a read",
"Unbound constructor F"]

(* A finding: an optional argument is applied to an OPTION, so [?arg_name:"pi"] never types, and
   this arm pins the reader's literal rule over a spelling no source can contain. The compilable
   spelling, [?arg_name:(Some "pi")], is read as a non-literal use (the non-literal cases' "optional
   application of an expression"). *)
[@@@known_rejection
"optional application with a literal is a read", "but an expression was expected of type"]
