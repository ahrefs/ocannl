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
  val alloc_zero_init_array : Ops.prec -> dims:int array -> stream -> buffer_ptr
  val free_buffer : (stream -> buffer_ptr -> unit) option
end

(** For now, we only configure a backend with regard to how many streams it should suggest using
    (where applicable). *)
type config = Only_devices_parallel | For_parallel_copying | Most_parallel_streams
[@@deriving equal, sexp, variants]

type merge_buffer_use = No | Streaming_for of Task.t | Copy [@@deriving sexp_of]

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

module type Device_config = sig
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

type ('buffer_ptr, 'dev, 'runner, 'event) device_ref = {
  dev : 'dev;
  ordinal : int;
  cross_stream_candidates : 'buffer_ptr Hashtbl.M(Tnode).t;
  owner_stream : ('buffer_ptr, 'dev, 'runner, 'event) stream_ref Hashtbl.M(Tnode).t;
  shared_writer_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
  host_reading_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
  host_writing_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
  mutable streams : ('buffer_ptr, 'dev, 'runner, 'event) stream_ref Utils.weak_dynarray;
}

and ('buffer_ptr, 'dev, 'runner, 'event) stream_ref = {
  device : ('buffer_ptr, 'dev, 'runner, 'event) device_ref;
  runner : 'runner;
  merge_buffer : 'buffer_ptr buffer option ref;
  stream_id : int;
  mutable allocated_buffer : 'buffer_ptr buffer option;
  updating_for : 'event Hashtbl.M(Tnode).t;
  mutable updating_for_merge_buffer : (Tnode.t * 'event option) option;
  reader_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
}

let sexp_of_device_ref _ _ _ _ device = [%sexp_of: string * int] ("ordinal", device.ordinal)
let sexp_of_stream_ref _ _ _ _ stream = [%sexp_of: string * int] ("stream_id", stream.stream_id)
let equal_stream_ref s1 s2 = s1.stream_id = s2.stream_id && s1.device.ordinal = s2.device.ordinal

type ('buffer_ptr, 'dev, 'runner, 'event) device =
      ('buffer_ptr, 'dev, 'runner, 'event) device_ref = {
  dev : 'dev;
  ordinal : int;
  cross_stream_candidates : 'buffer_ptr Hashtbl.M(Tnode).t;
      (** Freshly created arrays that might be shared across streams. The map can both grow and
          shrink. *)
  owner_stream : ('buffer_ptr, 'dev, 'runner, 'event) stream_ref Hashtbl.M(Tnode).t;
      (** The stream owning a given node. This map can only grow. Currently, if the memory mode of a
          node is inferred, only this stream will modify a cross-stream shared array. But memory
          modes can also be set manually. *)
  shared_writer_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
      (** The streams that most recently have been scheduled to update (write to) a
          cross-stream-shared node, and the associated update completion event. The completed events
          are removed opportunistically. *)
  host_reading_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
      (** The streams that most recently have been reading from a node's on-host array. The
          completed events are removed opportunistically. *)
  host_writing_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
      (** The streams that most recently have been writing to a node's on-host array. The completed
          events are removed opportunistically. *)
  mutable streams : ('buffer_ptr, 'dev, 'runner, 'event) stream_ref Utils.weak_dynarray;
      (** All (live) streams created on the device. Used by
          {!With_buffer_retrieval_and_syncing.sync_device}. Warning: stream_id fields of garbage
          collected streams can be reused! *)
}
[@@deriving sexp_of]

type ('buffer_ptr, 'dev, 'runner, 'event) stream =
      ('buffer_ptr, 'dev, 'runner, 'event) stream_ref = {
  device : ('buffer_ptr, 'dev, 'runner, 'event) device_ref;
  runner : 'runner;
  merge_buffer : 'buffer_ptr buffer option ref;
      (** Depending on backend implementations, either the currently used merge buffer, or the one
          most recently scheduled. Note that the pointer can be reused for nodes that fit in an
          already allocated buffer. *)
  stream_id : int;  (** An ID unique within the device for the lifetime of the stream. *)
  mutable allocated_buffer : 'buffer_ptr buffer option;
  updating_for : 'event Hashtbl.M(Tnode).t;
  (* The completion event for the most recent updating (writing to) a node via this stream. *)
  mutable updating_for_merge_buffer : (Tnode.t * 'event option) option;
      (** The tensor node that was most recently scheduled to be in the [stream]'s merge buffer. The
          event finishes after the [task] from a [Streaming_for task]. See also
          {!field-updating_for}. *)
  reader_streams :
    (('buffer_ptr, 'dev, 'runner, 'event) stream_ref * 'event) list Hashtbl.M(Tnode).t;
      (** The streams, other than this stream, that most recently have been reading from a node in
          this stream's context, and the associated use completion events. The completed events are
          removed opportunistically. *)
}
[@@deriving sexp_of]

let equal_stream = equal_stream_ref

type ('buffer_ptr, 'stream) context = {
  stream : 'stream;
  parent : ('buffer_ptr, 'stream) context option;
  ctx_arrays : 'buffer_ptr ctx_arrays;
      (** This map contains arrays used in this context or an ancestor context (they might be unique
          but might also be cross-stream shared. *)
  finalized : Utils.atomic_bool;
}
[@@deriving sexp_of]

module type Device_types = sig
  include Device_config

  type nonrec device = (buffer_ptr, dev, runner, event) device [@@deriving sexp_of]
  type nonrec stream = (buffer_ptr, dev, runner, event) stream [@@deriving sexp_of]
  type nonrec context = (buffer_ptr, stream) context [@@deriving sexp_of]
end

module type Device = sig
  include Device_types
  include Alloc_buffer with type buffer_ptr := buffer_ptr and type stream := stream

  val make_device : dev -> ordinal:int -> device
  val make_stream : device -> runner -> stream

  val make_context : ?ctx_arrays:ctx_arrays -> stream -> context
  (** Returns a context without a parent. *)

  val make_child : ?ctx_arrays:ctx_arrays -> context -> context
  (** Returns a context with the same {!field:stream}, and {!field:ctx_arrays} if omitted, as the
      given context's, which is also the {!field:parent}. *)

  val get_name : stream -> string
end

(** Parts shared by both assignments-level and lowered-level backend interfaces. *)
module type Backend_any_common = sig
  include Buffer

  val initialize : config -> unit
  (** Initializes a backend before first use. Typically does nothing if the backend is already
      initialized, but some backends can do some safe cleanups. *)

  val is_initialized : unit -> bool
  (** Returns false if there was no previous {!initialize} call. If it returns false, one must call
      {!initialize} before using the backend. *)
end

(** Parts shared by assignments-level backend interfaces. *)
module type Backend_common = sig
  include Backend_any_common

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val compile : ?name:string -> Indexing.unit_bindings -> Assignments.comp -> code
  (** [name] is used to derive names for compilation artifacts. If omitted, it's derived via
      {!Assignments.get_name_exn}. *)

  val compile_batch :
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
  include Backend_any_common with type buffer_ptr := buffer_ptr

  val sync : event -> unit
  (** Blocks till the event completes, if it's not done already.

      FIXME: it should rarely be needed to call [sync] explicitly, because it should always be
      called internally when necessary, in particular before extracting values from host. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's stream.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it should always
      be called internally when necessary. *)

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

  val suggested_num_streams : device -> int
  (** The optimal number of streams for the given device to follow the {!type:config} strategy. *)

  val new_stream : device -> stream
end

module type With_buffer_retrieval_and_syncing = sig
  type device
  type context
  type event

  val from_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy from host to context and
      returns true, otherwise returns false. *)

  val to_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy from context to host and
      returns true, otherwise returns false. *)

  val device_to_device :
    Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
  (** [device_to_device tn ~into_merge_buffer ~dst ~src] proceeds as follows:
      - If the node is absent from the [src] context and either it is present in the [dst] context
        or [into_merge_buffer] is different from [No]: raises an error.
      - If the node is absent from [dst] and [into_merge_buffer=No]: returns false.
      - Schedules waiting for writing into the tensor node on [src] to finish, if any.
      - If [into_merge_buffer=No]: schedules a copy of the tensor node from [src] to [dst] and
        updates the writer event for the node.
      - If [into_merge_buffer] is different from [No]: sets on [dst] the merge buffer source to the
        given node.
      - If [into_merge_buffer=Streaming_for task], remembers the buffer pointer of the source node
        to use for streaming, runs [task] -- intended to be the routine making use of the merge
        buffer, and initializes the merge buffer's streaming event.
      - If [into_merge_buffer=Copy], schedules copying from [src] to the merge buffer of [dst]'s
        stream, and updates the writer event for the merge buffer. *)

  val sync_device : device -> unit
  (** Synchronizes all the streams on a device, and cleans up (removes) all associated events. *)
end

module type Backend = sig
  include Backend_common
  include Backend_device_common with type buffer_ptr := buffer_ptr

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
