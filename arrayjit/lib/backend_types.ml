open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module No_device_types = struct
  type ctx_array = Ndarray.t [@@deriving sexp_of]

  type ctx_arrays = { used_memory : Utils.atomic_int; ctx_arrays : ctx_array Map.M(Tnode).t }
  [@@deriving sexp_of]

  let empty_ctx_arrays = { used_memory = Atomic.make 0; ctx_arrays = Map.empty (module Tnode) }
  let get_array arrays = Map.find arrays.ctx_arrays
end

module Types = struct
  type 'context routine = {
    context : 'context;
    schedule : Task.t;
    bindings : Indexing.lowered_bindings;
    name : string;
  }
  [@@deriving sexp_of]

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

module type Backend_common = sig
  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type buffer_ptr [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type routine = context Types.routine [@@deriving sexp_of]
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

  val get_used_memory : unit -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

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

module type No_device_backend = sig
  include Backend_common with type init_info := string and type stream := unit

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

  val to_buffer : Tnode.t -> dst:buffer_ptr -> src:context -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
  val get_buffer : Tnode.t -> context -> buffer_ptr option
end

module type Backend = sig
  type stream [@@deriving sexp_of]

  include Backend_common with type init_info := stream and type stream := stream

  val link : context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch : context -> code_batch -> context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  type event
  (** An event tracks if a stream finished computing past a particular point in its schedue. These
      values are used internally for scheduling across streams of the backend, and can be used for
      explicit scheduling. *)

  val sync : event -> unit
  (** Blocks till the event completes, if it's not done already. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val work_for : context -> Tnode.t -> event option
  (** If the tensor node is in the context, returns the event indicating if currently running or
      scheduled computations modifying that node on the context's stream have completed.

      NOTE: [work_for ctx tn], if work tracking was not yet registered for [tn], will register work
      tracking for [tn] and return the [all_work] event for [ctx]'s stream. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's stream.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it is typically
      called internally when necessary. But there is one exception, see {!device_to_device} when
      [into_merge_buffer=Streaming]. *)

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

  type device

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
  val to_subordinal : stream -> int
  val get_name : stream -> string
end

module type Lowered_backend_common = sig
  type context [@@deriving sexp_of]
  type ctx_array [@@deriving sexp_of]
  type ctx_arrays [@@deriving sexp_of]
  type buffer_ptr [@@deriving sexp_of]
  type config
  type init_info
  type stream

  val buffer_ptr : ctx_array -> buffer_ptr
  val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> stream -> buffer_ptr
  val ctx_arrays : context -> ctx_arrays
  val get_array : ctx_arrays -> Tnode.t -> ctx_array option

  val is_in_context : Low_level.traced_array -> bool
  (** If true, the node is required to be in the contexts linked with code that uses it.

      Should return false for nodes that are virtual, local, or which the backend prefers to access
      directly from the host. *)

  val initialize : config -> unit
  val is_initialized : unit -> bool
  val init : init_info -> context
  val finalize : context -> unit
  val name : string
end

module type Lowered_no_device_backend = sig
  include
    Lowered_backend_common
      with type stream := unit
       and type config := unit
       and type init_info := string

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

  val to_buffer : Tnode.t -> dst:buffer_ptr -> src:context -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

module type Lowered_backend = sig
  type stream [@@deriving sexp_of]

  include
    Lowered_backend_common
      with type config := Types.config
       and type stream := stream
       and type init_info := stream

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type event

  val sync : event -> unit
  val is_done : event -> bool
  val work_for : context -> Tnode.t -> event option
  val will_wait_for : context -> event -> unit

  open Types

  val sexp_of_context : context -> Sexplib.Sexp.t
  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    code_batch

  val link : context -> code -> context * Indexing.lowered_bindings * Task.t

  val link_batch :
    context -> code_batch -> context * Indexing.lowered_bindings * Task.t option array

  val from_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from host to context. *)

  val to_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from context to host. *)

  val device_to_device :
    Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
  (** See {!Backend.device_to_device}. *)

  type device

  val get_used_memory : device -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  val await : stream -> unit
  val is_idle : stream -> bool
  val all_work : stream -> event

  val scheduled_merge_node : stream -> Tnode.t option
  (** [scheduled_merge_node stream] is the tensor node that would be in the [stream]'s merge buffer
      right after [await stream]. *)

  val num_devices : unit -> int
  val suggested_num_streams : device -> int
  val get_device : ordinal:int -> device
  val get_stream_device : stream -> device
  val new_stream : device -> stream
  val get_ctx_stream : context -> stream
  val get_name : stream -> string
  val to_ordinal : device -> int
  val to_subordinal : stream -> int
end
