type t = Unix.process_status * string * string
(** Backend-free fresh-process probes. Output is captured in separate temporary files, so a child
    writing more than a pipe buffer to either stream cannot deadlock the parent. *)

val executable : unit -> string
(** Absolute path to this process's executable, resolved before changing directories. *)

val run : ?exe:string -> ?cwd:string -> ?temp_dir:string -> string list -> t
(** Run with inherited stdin and environment. [exe] defaults to [executable ()]; a relative [exe] is
    resolved against the parent's directory, before [cwd] takes effect. Capture files and file
    descriptors are cleaned on success and exceptions, and the parent's directory is restored. This
    is a synchronous direct-child runner; tests needing deadlines, environment overrides or
    concurrent process orchestration keep their own harness. *)

val output : t -> string
(** Stdout followed by stderr, for diagnostics whose stream does not matter. *)

val matches : ?stream:[ `Stdout | `Stderr | `Both ] -> exit:int -> contains:string list -> t -> bool
(** Require both the normal exit code and all causal diagnostic fragments. *)

val describe_status : Unix.process_status -> string

val prefixed : string -> string
(** Prefix every line of child text, including Verdict markers, before echoing it in a parent log.
*)

val report : label:string -> t -> unit
(** Print status and prefixed, separated streams to stderr when a parent's check fails. *)
