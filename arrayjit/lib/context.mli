(** Simplified context-based interface for backend operations *)

module Backends_deprecated = Backends

(** Execution context managing device, compilation, and buffers *)
type t

(** A compiled computational routine ready for execution *)
type routine

(** {2 Context creation} *)

(** Create a CUDA context. *)
val cuda : ?device_id:int -> unit -> t

(** Create a Metal context. *)  
val metal : ?device_id:int -> unit -> t

(** Create a CPU context. *)
val cpu : ?threads:int -> unit -> t

(** Automatically select the best available backend. *)
val auto : unit -> t

(** {2 Core operations} *)

(** Compile assignments into an executable routine.
    Returns updated context and the compiled routine. *)
val compile : t -> Ir.Assignments.comp -> Ir.Indexing.unit_bindings -> t * routine

(** Execute a compiled routine. Mutates buffers in-place. 
    Returns updated context with newly initialized nodes tracked. *)
val run : t -> routine -> t

(** {2 Data operations} *)

(** Note: These operations work with backend-specific buffer types hidden behind
    the context abstraction. For v0.6.1, you may need to use the existing 
    backend API for actual buffer manipulation. *)

(** Copy a tensor from source context to destination context. *)
val copy : src:t -> dst:t -> Ir.Tnode.t -> unit

(** {2 Node tracking operations} *)

(** Initialize a node from host memory (Ndarray/bigarray data).
    This is a temporary solution until the v0.7 refactoring removes hosted arrays.
    After calling this, the node is marked as initialized in the returned context. *)
val init_from_host_deprecated : t -> Ir.Tnode.t -> t

(** Check if a node is initialized. *)
val is_initialized : t -> Ir.Tnode.t -> bool

(** {2 Debug operations} *)

(** Get the name of the backend. *)
val backend_name : t -> string

(** Get the device ID. *)
val device_id : t -> int