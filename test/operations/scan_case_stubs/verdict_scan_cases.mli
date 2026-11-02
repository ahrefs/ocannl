(** The stand-in names the [{ocaml|...|ocaml}] snippets of [verdict_scan_cases.ml] call, for
    [tools/typecheck-scan-cases.sh], which compiles this interface first and opens it at the top of
    every snippet. The snippets exercise a reader of printed claims, so what matters is that each
    one is a program a compiler accepts; the types here are the least that make the intended reading
    of each call well-typed. *)

(** Printing functions, as the arms spell them: [Stdio]'s are the real formats, while a bare
    [printf] is the census arms' stand-in, applied to any arguments. *)
module Stdio : sig
  val printf : ('a, out_channel, unit) format -> 'a
  val eprintf : ('a, out_channel, unit) format -> 'a
end

val printf : string -> 'a

val report : string -> bool -> unit
(** A wrapper a claim is handed to, as the pre-[Verdict] tests defined for themselves. *)

val pass_fail : string -> bool -> unit
(** The dialect arms' verdict pair and the values they are applied to. *)

val skipped : string -> unit
val ok : bool
val other : string

val label : string
(** A [label] bound before the snippet: what the use preceding the snippet's own literal binding ("a
    later literal does not resolve an earlier use") reads in a program a compiler builds. *)

(** The skip-list arms (gh-ocannl-1794): [Base]'s collection iterators, the [Verdict] functions they
    spell qualified, the leg claim, and the names their gates, labels and populations read. *)
module List : sig
  val iter : 'a list -> f:('a -> unit) -> unit
  val filter_map : 'a list -> f:('a -> 'b option) -> 'b list
  val singleton : 'a -> 'a list
end

module Array : sig
  val iter : 'a array -> f:('a -> unit) -> unit
  val get : 'a array -> int -> 'a
end

module Option : sig
  val iter : 'a option -> f:('a -> unit) -> unit
end

module Verdict : sig
  type leg

  val skipped : backend:string -> string -> unit
  val leg : when_:bool -> on:string -> unit -> leg
end

val in_leg : leg:Verdict.leg -> string -> (string -> unit) -> unit
val p : string -> bool -> unit
val on_gpu : bool
val on_cpu : bool
val a : bool
val b : bool
val x : bool
val y : bool
val backend_name : string
val tag : string
val first : string
val second : string
val labels : string list
val names : string array
val maybe_label : string option
val cases : int list
val case_label : int -> string
val passes : int -> bool
val xs : int list
val skips : int list
val pair : int -> int option
val check : unit -> unit
val run : unit -> unit
val cap : [ `Withheld | `Advertised ]

(* The project-syntax arm is checked as a top-level item of a test that reports claims, under its
   stanza's preprocessors and libraries. It is a parse shape: it pins the scanner's reading of a
   claim inside an extension's payload, whose name no preprocessor expands -- any [[%name ...]]
   reads the same. *)
[@@@owning_module "test/gpt2/gpt2_dry_run.ml"]

[@@@known_rejection
"a claim inside an extension payload is reached", "Uninterpreted extension 'probe'"]
