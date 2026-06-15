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
    routine. The returned context carries the updated compilation frontier for dependency tracking;
    the input context is unchanged (see {!section:execution_deps}). *)

val run : t -> routine -> t
(** Execute a compiled routine. Mutates buffers in-place. Returns updated context with newly
    initialized nodes tracked. Raises [Failure] if execution dependencies are not satisfied. *)

(** {2 Execution dependency tracking}

    Execution dependencies mirror compilation dependencies: they record which routines must execute
    before which others based on tensor-node read/write hazards (RAW, WAR, WAW).

    Dependencies are scoped to compilation lineage: two routines compiled from the {i same}
    [Context.t] are independent siblings, even if they access the same nodes. Only routines compiled
    from the {i returned} (child) context of a prior [compile] call can depend on that prior
    routine. This matches how [compile] advances backend state only in the returned context. *)

val routine_id : routine -> int
(** A unique integer identifying the routine within its root context's lifetime. *)

val routine_name : routine -> string
(** The name of the routine, derived from the backend compilation. *)

val execution_deps : routine -> int list
(** The routine IDs that must execute before this routine, derived from RAW, WAR, and WAW hazards on
    tensor nodes at compile time. An empty list means the routine is independent of all previously
    compiled routines in its lineage. *)

val can_run : t -> routine -> bool
(** Whether all execution dependencies of the routine have been satisfied (i.e., all prerequisite
    routine IDs have been executed). *)

(** {2 Data operations} *)

(** Note: These operations work with backend-specific buffer types hidden behind the context
    abstraction. *)

val copy : src:t -> dst:t -> Ir.Tnode.t -> unit
(** Copy a tensor from source context to destination context. *)

(** {2 On-demand host access}

    After [gh-ocannl-333] no tensor data is stored on the host side of a tensor node. All CPU-side
    value access is an {b on-demand, context-mediated} device-to-host (or host-to-device) transfer
    through a temporary host buffer. There is no cache: every call performs a fresh transfer, which
    is {b expensive on non-unified-memory backends} — prefer batching over polling. *)

val to_host : t -> Ir.Tnode.t -> Ir.Ndarray.t
(** Transfers the node's device buffer into a fresh host [Ndarray] and returns it. Raises if the
    node is not present in the context. *)

val from_host : t -> Ir.Tnode.t -> Ir.Ndarray.t -> t
(** Uploads the host buffer into the node's device buffer (allocating it if needed) and returns a
    context in which the node is marked initialized. *)

val get_values : t -> Ir.Tnode.t -> float array
(** Retrieves all (unpadded) values of the node via an on-demand device-to-host transfer. *)

val set_values : t -> Ir.Tnode.t -> float array -> t
(** Sets all (unpadded) values of the node via an on-demand host-to-device transfer, returning a
    context in which the node is marked initialized. *)

val get_value : t -> Ir.Tnode.t -> int array -> float
(** Retrieves a single value at the given index via an on-demand device-to-host transfer. *)

val set_value : t -> Ir.Tnode.t -> int array -> float -> t
(** Sets a single value at the given index, preserving the other elements. Returns a context in
    which the node is marked initialized. *)

val points_1d : ?from_axis:int -> xdim:int -> t -> Ir.Tnode.t -> float array
(** Like {!get_values} but extracts a 1d slice of points for plotting. *)

val points_2d : ?from_axis:int -> xdim:int -> ydim:int -> t -> Ir.Tnode.t -> (float * float) array
(** Like {!get_values} but extracts a 2d slice of points for plotting. *)

(** {2 Node tracking operations} *)

val is_initialized : t -> Ir.Tnode.t -> bool
(** Check if a node is initialized. *)

(** {2 Debug operations} *)

val backend_name : t -> string
(** Get the name of the backend. *)

val device_id : t -> int
(** Get the device ID. *)
