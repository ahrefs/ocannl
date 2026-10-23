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
