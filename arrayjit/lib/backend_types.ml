open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module type Buffer_ptr = sig
  type buffer_ptr [@@deriving sexp_of]
  type ctx_arrays = buffer_ptr Map.M(Tnode).t [@@deriving sexp_of]
end

module No_device_buffer_ptr : Buffer_ptr with type buffer_ptr = Ndarray.t = struct
  type buffer_ptr = Ndarray.t [@@deriving sexp_of]
  type ctx_arrays = buffer_ptr Map.M(Tnode).t [@@deriving sexp_of]
end

module Types = struct
  type 'context routine = {
    context : 'context;
    schedule : Task.t;
    bindings : Indexing.lowered_bindings;
    name : string;
    inputs : Set.M(Tnode).t;
        (** The materialized read-only and read-before-write (within the routine) non-constant
            nodes. They are inputs in a broad sense, as they could be recurrent nodes or parameters. *)
    outputs : Set.M(Tnode).t;  (** All the materialized nodes written-to by the routine. *)
  }
  [@@deriving sexp_of]

  type ('buffer_ptr, 'event, 'device, 'state, 'runner) stream = {
    device : 'device;
    state : 'state;
    merge_buffer : ('buffer_ptr * Tnode.t) option ref;
    unique_name : string;
    mutable allocated_buffer : ('buffer_ptr * int) option;
    runner : 'runner;
    requested_work_for : 'event option Hashtbl.M(Tnode).t;
  }
  [@@deriving sexp_of]

  let make_stream ~device ~state ~unique_name ~runner =
    {
      device;
      state;
      merge_buffer = ref None;
      unique_name : string;
      allocated_buffer = None;
      runner : 'runner;
      requested_work_for = Hashtbl.create (module Tnode);
    }

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
end

(** Parts shared by both assignments-level and lowered-level backend interfaces. *)
module type Backend_any_common = sig
  type buffer_ptr [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type stream

  type init_info
  (** For backends derived via {!No_device_backend}, this is usually the backend name concatenated
      with the device or stream number. For {!Backend}, [init_info = stream]. *)

  val name : string

  val initialize : Types.config -> unit
  (** Initializes a backend before first use. Typically does nothing if the backend is already
      initialized, but some backends can do some safe cleanups. *)

  val is_initialized : unit -> bool
  (** Returns false if there was no previous {!initialize} call. If it returns false, one must call
      {!initialize} before using the backend. *)

  val init : init_info -> context

  val finalize : context -> unit
  (** Finalizes (just) the context. *)

  val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> stream -> buffer_ptr
end

(** Parts shared by assignments-level backend interfaces. *)
module type Backend_common = sig
  include Backend_any_common

  type routine = context Types.routine [@@deriving sexp_of]
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

(** Parts shared by backend implementations excluding what's already in {!Backend_any_common}. *)
module type Backend_impl_common = sig
  type context [@@deriving sexp_of]

  include Buffer_ptr

  val ctx_arrays : context -> ctx_arrays

  val is_in_context : Low_level.traced_array -> bool
  (** If true, the node is required to be in the contexts linked with code that uses it.

      Should return false for nodes that are virtual, local, or which the backend prefers to access
      directly from the host. *)
end

module type No_device_copying = sig
  type buffer_ptr

  val buffer_to_buffer : dst:buffer_ptr -> src:buffer_ptr -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

(** An intermediate interface for stream-agnostic (typically CPU) backend implementations. *)
module type No_device_backend = sig
  include Backend_common with type init_info := string and type stream := unit
  include Backend_impl_common with type context := context and type buffer_ptr := buffer_ptr

  val link : merge_buffer:(buffer_ptr * Tnode.t) option ref -> context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch :
    merge_buffer:(buffer_ptr * Tnode.t) option ref ->
    context ->
    code_batch ->
    context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines (in particular, the routines' contexts are not
      independent). *)

  val get_used_memory : unit -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  include No_device_copying with type buffer_ptr := buffer_ptr
end

(** Parts shared by both assignments-level and lowered-level backend interfaces providing streams
    and devices. *)
module type Backend_device_common = sig
  type buffer_ptr
  type device

  type event
  (** An event tracks if a stream finished computing past a particular point in its schedue. These
      values are used internally for scheduling across streams of the backend, and can be used for
      explicit scheduling. *)

  type stream_state [@@deriving sexp_of]
  type runner [@@deriving sexp_of]
  type stream = (buffer_ptr, event, device, stream_state, runner) Types.stream [@@deriving sexp_of]

  include
    Backend_any_common
      with type buffer_ptr := buffer_ptr
       and type init_info := stream
       and type stream := stream

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
  (** The optimal number of streams for the given device to follow the {!Types.config} strategy
      passed to {!No_device_backend.initialize}. *)

  val new_stream : device -> stream
  val get_ctx_stream : context -> stream
  val get_stream_device : stream -> device
  val to_ordinal : device -> int
  val get_name : stream -> string
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
    Tnode.t -> into_merge_buffer:Types.merge_buffer_use -> dst:context -> src:context -> bool
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
  include Backend_device_common

  include
    Backend_common
      with type context := context
       and type init_info := stream
       and type stream := stream

  val link : context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch : context -> code_batch -> context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  include With_buffer_retrieval_and_syncing with type context := context and type event := event
end

(** Lowered-level stream agnostic backend interface: implementation-facing API for CPU backends. *)
module type Lowered_no_device_backend = sig
  include Backend_impl_common

  include
    Backend_any_common
      with type context := context
       and type stream := unit
       and type init_info := string
       and type buffer_ptr := buffer_ptr

  type procedure [@@deriving sexp_of]

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized ->
    procedure

  val compile_batch :
    names:string option array ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    ctx_arrays option * procedure option array

  val link_compiled :
    merge_buffer:(buffer_ptr * Tnode.t) option ref ->
    context ->
    procedure ->
    context * Indexing.lowered_bindings * Task.t * string

  include No_device_copying with type buffer_ptr := buffer_ptr
end

module type No_buffer_retrieval_or_syncing = sig
  include Backend_impl_common
  include Backend_device_common with type context := context and type buffer_ptr := buffer_ptr

  val from_host : dst_ptr:buffer_ptr -> dst:context -> Ndarray.t -> unit
  (** Like {!Backend.from_host}, but without synchronization and buffer retrieval. *)

  val to_host : src_ptr:buffer_ptr -> src:context -> Ndarray.t -> unit
  (** Like {!Backend.to_host}, but without synchronization and buffer retrieval. *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:Types.merge_buffer_use ->
    dst_ptr:buffer_ptr option ->
    dst:context ->
    src_ptr:buffer_ptr ->
    src:context ->
    unit
  (** Like {!Backend.device_to_device}, but without synchronization and buffer retrieval. Raises
      [Invalid_argument] if [into_merge_buffer = No] and [dst_ptr = None]. *)
end

(** Lowered-level backend interface: implementation-facing API for device-based (typically GPU)
    backends. *)
module type Lowered_backend = sig
  include No_buffer_retrieval_or_syncing

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    code_batch

  val link : context -> code -> context * Indexing.lowered_bindings * Task.t

  val link_batch :
    context -> code_batch -> context * Indexing.lowered_bindings * Task.t option array

  val scheduled_merge_node : stream -> Tnode.t option
  (** [scheduled_merge_node stream] is the tensor node that would be in the [stream]'s merge buffer
      right after [await stream]. *)
end
