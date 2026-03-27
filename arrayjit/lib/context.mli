(** Simplified context-based interface for backend operations *)

open Base
module Backends_deprecated = Backends

type t [@@deriving sexp_of]
(** Execution context managing device, compilation, and buffers *)

type routine
(** A compiled computational routine ready for execution *)

val bindings : routine -> Ir.Indexing.lowered_bindings
val context : routine -> t

(** {2 Context creation} *)

val cuda : ?device_id:int -> unit -> t
(** Create a CUDA context. *)

val metal : ?device_id:int -> unit -> t
(** Create a Metal context. *)

val cpu : ?threads:int -> unit -> t
(** Create a CPU context. *)

val auto : unit -> t
(** Automatically select the best available backend. *)

(** {2 Core operations} *)

val compile : t -> Ir.Assignments.comp -> Ir.Indexing.unit_bindings -> t * routine
(** Compile assignments into an executable routine. Returns updated context and the compiled
    routine. The returned context carries the updated compilation frontier for dependency
    tracking; the input context is unchanged (see {!section:execution_deps}). *)

val run : t -> routine -> t
(** Execute a compiled routine. Mutates buffers in-place. Returns updated context with newly
    initialized nodes tracked. Raises [Failure] if execution dependencies are not satisfied. *)

(** {2 Execution dependency tracking}

    Execution dependencies mirror compilation dependencies: they record which routines must execute
    before which others based on tensor-node read/write hazards (RAW, WAR, WAW).

    Dependencies are scoped to compilation lineage: two routines compiled from the {i same}
    [Context.t] are independent siblings, even if they access the same nodes. Only routines
    compiled from the {i returned} (child) context of a prior [compile] call can depend on that
    prior routine. This matches how [compile] advances backend state only in the returned context.
*)

val routine_id : routine -> int
(** A unique integer identifying the routine within its root context's lifetime. *)

val routine_name : routine -> string
(** The name of the routine, derived from the backend compilation. *)

val execution_deps : routine -> int list
(** The routine IDs that must execute before this routine, derived from RAW, WAR, and WAW hazards
    on tensor nodes at compile time. An empty list means the routine is independent of all
    previously compiled routines in its lineage. *)

val can_run : t -> routine -> bool
(** Whether all execution dependencies of the routine have been satisfied (i.e., all prerequisite
    routine IDs have been executed). *)

(** {2 Data operations} *)

(** Note: These operations work with backend-specific buffer types hidden behind the context
    abstraction. For v0.6.1, you may need to use the existing backend API for actual buffer
    manipulation. *)

val copy : src:t -> dst:t -> Ir.Tnode.t -> unit
(** Copy a tensor from source context to destination context. *)

(** {2 Node tracking operations} *)

val init_from_host_deprecated : t -> Ir.Tnode.t -> t
(** Initialize a node from host memory (Ndarray/bigarray data). This is a temporary solution until
    the v0.7 refactoring removes hosted arrays. After calling this, the node is marked as
    initialized in the returned context. *)

val is_initialized : t -> Ir.Tnode.t -> bool
(** Check if a node is initialized. *)

(** {2 Debug operations} *)

val backend_name : t -> string
(** Get the name of the backend. *)

val device_id : t -> int
(** Get the device ID. *)
