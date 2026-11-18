(** Raw CUDA Driver API snapshots for the opt-in reservation witness. The current CUDA device must
    be set, and the exclusive run checks for external compute processes. *)

type snapshot

val read : unit -> snapshot
val report : ?baseline:snapshot -> string -> snapshot -> unit
val after_first_exec_destroy : (unit -> 'a) -> 'a * snapshot option
val streams : unit -> unit
