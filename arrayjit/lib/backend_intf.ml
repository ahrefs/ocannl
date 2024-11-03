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

  val c_ptr_to_string : (buffer_ptr -> Ops.prec -> string) option

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

type merge_buffer_use = No | Streaming | Copy [@@deriving equal, sexp]

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

type ('buffer_ptr, 'dev, 'event) device = {
  dev : 'dev;
  ordinal : int;
  mutable shared_merge_buffer : 'buffer_ptr buffer option;
  mutable latest_stream_id : int;
  released : Utils.atomic_bool;
  cross_stream_candidates : 'buffer_ptr Hashtbl.M(Tnode).t;
      (** Freshly created arrays that might be shared across streams. The map can both grow and
          shrink. See the explanation on top of this file. *)
  owner_streams : int Hashtbl.M(Tnode).t;
      (** The streams owning the given nodes. This map can only grow. *)
  stream_working_on : (int * 'event) option Hashtbl.M(Tnode).t;
      (** The stream that most recently has been updating the node, and the associated update
          completion event. An entry for a tensor node is only populated when
          {!field-queried_work_for} is also populated. *)
}
[@@deriving sexp_of]

type ('buffer_ptr, 'dev, 'runner, 'event) stream = {
  device : ('buffer_ptr, 'dev, 'event) device;
  runner : 'runner;
  merge_buffer : ('buffer_ptr * Tnode.t) option ref;
  stream_id : int;
  mutable allocated_buffer : 'buffer_ptr buffer option;
  queried_work_for : 'event option Hashtbl.M(Tnode).t;
      (* The completion event for updating the node via this stream, if any. Only existing entries
         are updated, and an entry is populated when {!work_for} is called for the first time on the
         tensor node. *)
}
[@@deriving sexp_of]

(** [scheduled_merge_node stream] is the tensor node that would be in the [stream]'s merge buffer
    right after [await stream]. *)
let scheduled_merge_node stream = Option.map ~f:snd !(stream.merge_buffer)

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

  type nonrec device = (buffer_ptr, dev, event) device [@@deriving sexp_of]
  type nonrec stream = (buffer_ptr, dev, runner, event) stream [@@deriving sexp_of]
  type nonrec context = (buffer_ptr, stream) context [@@deriving sexp_of]
end

module type Device = sig
  include Device_types
  include Alloc_buffer with type buffer_ptr := buffer_ptr and type stream := stream

  val make_device : dev -> ordinal:int -> device
  val make_stream : device -> runner -> stream_id:int -> stream

  val make_context : ?ctx_arrays:ctx_arrays -> stream -> context
  (** Returns a context without a parent. *)

  val make_child : ?ctx_arrays:ctx_arrays -> context -> context
  (** Returns a context with the same {!field-stream}, and {!field-ctx_arrays} if omitted, as the
      given context's, which is also the {!field-parent}. *)

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

  val compile : ?shared:bool -> ?name:string -> Indexing.unit_bindings -> Assignments.comp -> code
  (** If [~shared:true] (default [false]), the backend should prefer to do more compile work in a
      device-and-stream-agnostic way. If [~shared:false], the backend can opt to postpone compiling
      altogether until [link] is called, to benefit from more optimizations. *)

  val compile_batch :
    ?shared:bool ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.comp array ->
    code_batch
  (** Unlike the [~shared] parameter, [compile_batch] vs. [compile] is mostly about improving the
      compile time and debugging convenience by generating fewer files -- ideally does not affect
      execution, but there can be backend-specific differences. Only array entries for which
      [occupancy] returns true are included. *)
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
  (** Blocks till the event completes, if it's not done already. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's stream.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it is typically
      called internally when necessary. But there is one exception, see {!device_to_device} when
      [into_merge_buffer=Streaming]. *)

  val get_used_memory : device -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

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
  (** The optimal number of streams for the given device to follow the {!config} strategy. *)

  val new_stream : device -> stream
end

module type With_buffer_retrieval_and_syncing = sig
  type context
  type event

  val work_for : context -> Tnode.t -> event option
  (** If the tensor node is in the context, returns the event indicating if currently running or
      scheduled computations modifying that node on the context's stream have completed.

      NOTE: [work_for ctx tn], if work tracking was not yet registered for [tn], will register work
      tracking for [tn] and return the [all_work] event for [ctx]'s stream. *)

  val from_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy from host to context and
      returns true, otherwise returns false. NOTE: it's the caller's responsibility to synchronize
      the stream (via [await ctx.stream] or [sync (work_for ctx tn)]) before the host's data is
      overwritten. *)

  val to_host : context -> Tnode.t -> bool
  (** If the tensor node is both hosted and in-context, schedules a copy from context to host and
      returns true, otherwise returns false. NOTE: it's the caller's responsibility to synchronize
      the stream (via [await ctx.stream] or [sync (work_for ctx tn)]) before the host's data is
      read. *)

  val device_to_device :
    Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
  (** [device_to_device tn ~into_merge_buffer ~dst ~src] proceeds as follows:
      - If the node is absent from the [src] context and either it is present in the [dst] context
        or [into_merge_buffer] is different from [No]: raises an error.
      - If the node is absent from [dst] and [into_merge_buffer=No]: returns false.
      - Executes [will_wait_for dst (work_for src tn)].
      - If [into_merge_buffer=No]: schedules a copy of the tensor node from [src] to [dst].
      - If [into_merge_buffer] is different from [No]: sets on [dst] the merge buffer source to the
        given node. If [into_merge_buffer=Streaming], remembers the buffer pointer of the source
        node to use for streaming, without blocking. If [into_merge_buffer=Copy], schedules copying
        from [src] to the merge buffer of [dst]'s stream.

      NOTE: If [into_merge_buffer=Streaming], after scheduling the work on [dst] using the merge
      buffer but before scheduling work on [src] that modifies [tn], execute
      [will_wait_for src (all_work (get_ctx_stream dst))]. *)
end

module type Backend = sig
  include Backend_common
  include Backend_device_common with type buffer_ptr := buffer_ptr

  val link : context -> code -> context routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch : context -> code_batch -> context * context routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  include With_buffer_retrieval_and_syncing with type context := context and type event := event
end
