(** {1 The interface types for backends}

    User-facing backend API. *)

open Base

type 'buffer_ptr buffer = { ptr : 'buffer_ptr; size_in_bytes : int } [@@deriving sexp_of]
type 'buffer_ptr ctx_arrays = 'buffer_ptr Map.M(Tnode).t [@@deriving sexp_of]

module Buffer_types (Buffer_ptr : sig
  type buffer_ptr [@@deriving sexp_of]
end) =
struct
  type nonrec buffer = Buffer_ptr.buffer_ptr buffer [@@deriving sexp_of]
  type nonrec ctx_arrays = Buffer_ptr.buffer_ptr ctx_arrays [@@deriving sexp_of]
end

module type Buffer = sig
  type buffer_ptr [@@deriving sexp_of]

  include module type of Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module type Alloc_buffer = sig
  include Buffer

  type stream

  val alloc_buffer : ?old_buffer:buffer -> size_in_bytes:int -> stream -> buffer
  val alloc_array : Ops.prec -> dims:int array -> stream -> buffer_ptr
  val alloc_zeros : Ops.prec -> dims:int array -> stream -> buffer_ptr
  val free_buffer : (stream -> buffer_ptr -> unit) option
end

type merge_buffer_use = No | Copy [@@deriving sexp_of]

type param_source =
  | Log_file_name
  | Merge_buffer
  | Param_ptr of Tnode.t
  | Static_idx of Indexing.static_symbol
[@@deriving sexp_of]

type 'context routine = {
  context : 'context;
  schedule : Task.t;
  bindings : Indexing.lowered_bindings;
  name : string;
  inputs : Set.M(Tnode).t;
      (** The materialized read-only and read-before-write (within the routine) non-constant nodes.
          They are inputs in a broad sense, as they could be recurrent nodes or parameters. *)
  merge_buffer_input : Tnode.t option;  (** Similar to {!field-inputs}, for the merge buffer. *)
  outputs : Set.M(Tnode).t;  (** All the materialized nodes written-to by the routine. *)
}
[@@deriving sexp_of]

module type Device_config_common = sig
  include Buffer

  type dev [@@deriving sexp_of]
  (** Interface to a device driver. *)

  type runner [@@deriving sexp_of]
  (** Interface to a stream driver. *)

  type event [@@deriving sexp_of]
  (** An event tracks if a stream finished computing past a particular point in its schedue. These
      values are used internally for scheduling across streams of the backend, and can be used for
      explicit scheduling. *)

  val name : string
end

module type Device_config = sig
  include Device_config_common

  type optimize_ctx [@@deriving sexp_of]
  (** The optimization context for compiling code, in particular {!Low_level.optimize_ctx} for
      low-level backends. *)

  val empty_optimize_ctx : unit -> optimize_ctx
end

type ('buffer_ptr, 'dev, 'runner, 'event) device_ref = {
  dev : 'dev;
  ordinal : int;
  device_id : int;
  device_buffer_cache : 'buffer_ptr Hashtbl.M(Tnode).t;
      (** Per-device buffer cache for reusing allocated arrays (e.g. read-only/constant nodes, or
          host-backed buffers on unified memory systems). *)
  mutable current_stream : ('buffer_ptr, 'dev, 'runner, 'event) stream_ref option;
  mutable next_stream_id : int;
}

and ('buffer_ptr, 'dev, 'runner, 'event) stream_ref = {
  device : ('buffer_ptr, 'dev, 'runner, 'event) device_ref;
  runner : 'runner;
  merge_buffer : 'buffer_ptr buffer option ref;
  stream_id : int;
  mutable allocated_buffer : 'buffer_ptr buffer option;
  merge_buffer_node : Tnode.t option ref;
      (** The tensor node currently occupying this stream's merge buffer, if any. Used for
          consistency checking before routine execution. *)
}

let sexp_of_device_ref _ _ _ _ device = [%sexp_of: string * int] ("ordinal", device.ordinal)
let sexp_of_stream_ref _ _ _ _ stream = [%sexp_of: string * int] ("stream_id", stream.stream_id)
let equal_stream_ref s1 s2 = s1.stream_id = s2.stream_id && s1.device.ordinal = s2.device.ordinal

type ('buffer_ptr, 'dev, 'runner, 'event) device =
  ('buffer_ptr, 'dev, 'runner, 'event) device_ref
[@@deriving sexp_of]

type ('buffer_ptr, 'dev, 'runner, 'event) stream =
  ('buffer_ptr, 'dev, 'runner, 'event) stream_ref
[@@deriving sexp_of]

let equal_stream = equal_stream_ref

type ('buffer_ptr, 'stream, 'optimize_ctx) context = {
  stream : 'stream;
  parent : ('buffer_ptr, 'stream, 'optimize_ctx) context option;
  ctx_arrays : 'buffer_ptr ctx_arrays;
      (** This map contains arrays used in this context or an ancestor context (they might be unique
          but might also be cross-stream shared). *)
  finalized : Utils.atomic_bool;
  optimize_ctx : 'optimize_ctx;
}
[@@deriving sexp_of]

module type Device_types = sig
  include Device_config

  type nonrec device = (buffer_ptr, dev, runner, event) device [@@deriving sexp_of]
  type nonrec stream = (buffer_ptr, dev, runner, event) stream [@@deriving sexp_of]
  type nonrec context = (buffer_ptr, stream, optimize_ctx) context [@@deriving sexp_of]
end

module type Device = sig
  include Device_types
  include Alloc_buffer with type buffer_ptr := buffer_ptr and type stream := stream

  val make_device : dev -> ordinal:int -> device
  val make_stream : device -> runner -> stream

  val make_context : ?ctx_arrays:ctx_arrays -> ?optimize_ctx:optimize_ctx -> stream -> context
  (** Returns a context without a parent. *)

  val make_child : ?ctx_arrays:ctx_arrays -> ?optimize_ctx:optimize_ctx -> context -> context
  (** Returns a context with the same {!field:Backend_intf.context.stream}, and
      {!field:Backend_intf.context.ctx_arrays}, {!field:Backend_intf.context.optimize_ctx} if
      omitted, as the given context's, which is also the {!field:Backend_intf.context.parent}. *)

  val get_name : stream -> string
end

(** Parts shared by assignments-level backend interfaces. *)
module type Backend_common = sig
  include Buffer

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type optimize_ctx [@@deriving sexp_of]

  val empty_optimize_ctx : unit -> optimize_ctx
  val get_optimize_ctx : code -> optimize_ctx
  val get_optimize_ctx_batch : code_batch -> optimize_ctx

  val compile : optimize_ctx -> ?name:string -> Indexing.unit_bindings -> Assignments.comp -> code
  (** [name] is used to derive names for compilation artifacts. If omitted, it's derived via
      {!Assignments.get_name_exn}. *)

  val compile_batch :
    optimize_ctx ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.comp array ->
    code_batch
  (** [compile_batch] vs. [compile] is mostly about improving the compile time and debugging
      convenience by generating fewer files -- ideally does not affect execution, but there can be
      backend-specific differences. Only array entries for which [occupancy] returns true are
      included. [names] are used to derive names for compilation artifacts. If omitted, they're
      derived via {!Assignments.get_name_exn}. *)
end

(** Parts shared by both assignments-level and lowered-level backend interfaces providing streams
    and devices, both user-facing and implementation-facing. Does not include: compilation and
    linking (differnt for assignments-level and lowered-level); copying and tensor-node-level
    synchronization (copying is different for user-facing and implementation-facing APIs,
    synchronization is provided by a component outside of backend implementations). *)
module type Backend_device_common = sig
  include Device

  val sync : event -> unit
  (** Blocks till the event completes, if it's not done already.

      It is rarely needed to call [sync] explicitly, because it should always be called internally
      when necessary, in particular before extracting values from host. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's stream.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it should always
      be called internally when necessary. *)

  val static_properties : Sexp.t
  (** Returns a sexp description of the properties of all devices. *)

  val get_used_memory : device -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  val get_global_debug_info : unit -> Sexp.t
  (** Global debug information; backend-specific and might evolve independently on the backends. *)

  val get_debug_info : stream -> Sexp.t
  (** Per-stream debug information; backend-specific and might evolve independently on the backends
  *)

  val await : stream -> unit
  (** Blocks till the stream becomes idle, i.e. synchronizes the stream. *)

  val all_work : stream -> event
  (** Returns the event indicating if any currently running or scheduled computations on the stream
      have completed. *)

  val is_idle : stream -> bool
  (** Whether the stream is currently waiting for work. *)

  val get_device : ordinal:int -> device
  val num_devices : unit -> int
  val new_stream : device -> stream
end

module type With_buffer_retrieval_and_syncing = sig
  type device
  type context
  type event

  val from_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy(^) from host to context and
      returns true, otherwise returns false.

      [^] On unified memory devices, the copy is not scheduled if the source and destination are the
      same buffer (note that this depends on the memory mode of the tensor node). *)

  val init_from_host : context -> Tnode.t -> context
  (** Schedules a copy from host to context: a variant of {!from_host} that requires the input
      context to not contain the tensor node, and outputs the context with the tensor node. *)

  val to_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy(^) from context to host and
      returns true, otherwise returns false.

      [^] On unified memory devices, the copy is not scheduled if the source and destination are the
      same buffer (note that this depends on the memory mode of the tensor node). *)

  val device_to_device :
    Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
  (** [device_to_device tn ~into_merge_buffer ~dst ~src] proceeds as follows:
      - If the node is absent from the [src] context and either it is present in the [dst] context
        or [into_merge_buffer] is different from [No]: raises an error.
      - If the node is absent from [dst] and [into_merge_buffer=No]: returns false.
      - If [into_merge_buffer=No]: schedules a copy of the tensor node from [src] to [dst] and
        updates the writer event for the node.
      - If [into_merge_buffer=Copy], schedules copying from [src] to the merge buffer of [dst]'s
        stream, and updates the writer event for the merge buffer. *)

  val init_from_device : Tnode.t -> dst:context -> src:context -> context
  (** Schedules a copy from [src] to [dst]: a variant of {!device_to_device} with
      [into_merge_buffer=No] that requires the input [src] context to not contain the tensor node,
      and outputs the [dst] context with the tensor node. *)

  val sync_device : device -> unit
  (** Synchronizes the device's stream and cleans up merge buffer state. *)
end

module type Backend = sig
  include Backend_common

  include
    Backend_device_common with type buffer_ptr := buffer_ptr and type optimize_ctx := optimize_ctx

  val link : context -> code -> context routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context.
  *)

  val link_batch : context -> code_batch -> context * context routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  include
    With_buffer_retrieval_and_syncing
      with type device := device
       and type context := context
       and type event := event
end
