type t = Unix.process_status * string * string
(** Backend-free fresh-process probes. Output is captured in separate temporary files, so a child
    writing more than a pipe buffer to either stream cannot deadlock the parent. *)

val executable : unit -> string
(** Absolute path to this process's executable, resolved before changing directories. *)

val nested_dune_environment : ?env:string array -> unit -> string array
(** [env] (default: this process's environment) without the variables by which Dune tells a child it
    runs as one of Dune's actions ([INSIDE_DUNE], [DUNE_ACTION_TRACE_DIR]). A test that launches its
    own, independent Dune build from inside a Dune action passes this as [run]'s [env]: handed them,
    the nested Dune resolves paths as the parent action and writes its trace into the parent's
    action-trace directory. [explicit_library_deps] pins both effects. *)

val run :
  ?exe:string ->
  ?cwd:string ->
  ?temp_dir:string ->
  ?env:string array ->
  ?stderr:[ `Capture | `Discard | `Into of Unix.file_descr ] ->
  string list ->
  t
(** Run with inherited stdin, and with [env] or else the inherited environment. [exe] defaults to
    [executable ()]; a relative [exe] is resolved against the parent's directory, before [cwd] takes
    effect. Stdout is always captured. [stderr] defaults to [`Capture]; [`Discard] sends the child's
    stderr to the null device, and [`Into fd] to the caller's [fd], which stays the caller's to
    close. With either, the result's stderr is empty. Capture files and the descriptors [run] opens
    are cleaned on success and exceptions, and the parent's directory is restored. This is a
    synchronous direct-child runner; tests needing deadlines or concurrent process orchestration
    keep their own harness. *)

val output : t -> string
(** Stdout followed by stderr, for diagnostics whose stream does not matter. *)

val matches : ?stream:[ `Stdout | `Stderr | `Both ] -> exit:int -> contains:string list -> t -> bool
(** Require both the normal exit code and all causal diagnostic fragments. Each fragment must occur
    wholly within one captured stream; [`Both] allows different fragments in either stream. *)

val describe_status : Unix.process_status -> string

val prefixed : string -> string
(** Prefix every line of child text, including Verdict markers, before echoing it in a parent log.
*)

val report : label:string -> t -> unit
(** Print status and prefixed, separated streams to stderr when a parent's check fails. *)
