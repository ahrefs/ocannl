(** {1 The components for use in backend implementations}

    Implementation-facing types and components. *)

open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

open Backend_intf

module type No_device_buffer_and_copying = sig
  include Alloc_buffer with type stream := unit

  val get_used_memory : unit -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  val buffer_to_buffer : dst:buffer_ptr -> src:buffer_ptr -> size_in_bytes:int -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

module No_device_buffer_and_copying () :
  No_device_buffer_and_copying with type buffer_ptr = unit Ctypes.ptr = struct
  type buffer_ptr = unit Ctypes.ptr

  let sexp_of_buffer_ptr = Ops.sexp_of_voidptr

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)

  let used_memory = Atomic.make 0
  let get_used_memory () = Atomic.get used_memory

  let alloc_impl ~size_in_bytes =
    let finalize _ptr = ignore (Atomic.fetch_and_add used_memory ~-size_in_bytes : int) in
    let ptr = Ctypes.(to_voidp @@ allocate_n int8_t ~count:size_in_bytes) in
    let _ : int = Atomic.fetch_and_add used_memory size_in_bytes in
    Stdlib.Gc.finalise finalize ptr;
    ptr

  let alloc_zero_init_array prec ~dims () =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    alloc_impl ~size_in_bytes

  let alloc_buffer ?old_buffer ~size_in_bytes () =
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | _ -> { ptr = alloc_impl ~size_in_bytes; size_in_bytes }

  let free_buffer = None

  let buffer_to_buffer ~dst:Ctypes_static.(CPointer dst) ~src:Ctypes_static.(CPointer src)
      ~size_in_bytes =
    Ctypes_memory_stubs.memcpy ~dst ~src ~size:size_in_bytes

  let host_to_buffer src ~dst:Ctypes_static.(CPointer dst) =
    Ctypes_memory_stubs.memcpy ~dst
      ~src:(Ndarray.get_fatptr_not_managed src)
      ~size:(Ndarray.size_in_bytes src)

  let buffer_to_host dst ~src:Ctypes_static.(CPointer src) =
    Ctypes_memory_stubs.memcpy
      ~dst:(Ndarray.get_fatptr_not_managed dst)
      ~src ~size:(Ndarray.size_in_bytes dst)

  let c_ptr_to_string = Some Ops.c_ptr_to_string
end

module Device_types (Device_config : Device_config) = struct
  include Device_config

  type nonrec device =
    (Device_config.buffer_ptr, Device_config.dev, Device_config.runner, Device_config.event) device
  [@@deriving sexp_of]

  type nonrec stream =
    (Device_config.buffer_ptr, Device_config.dev, Device_config.runner, Device_config.event) stream
  [@@deriving sexp_of]

  type nonrec context = (buffer_ptr, stream) context [@@deriving sexp_of]
end

module Device
    (Device_types : Device_types)
    (Alloc_buffer : Alloc_buffer
                      with type buffer_ptr := Device_types.buffer_ptr
                       and type stream := Device_types.stream) =
struct
  include Device_types
  include Alloc_buffer

  let make_device dev ~ordinal =
    {
      dev;
      ordinal;
      shared_merge_buffer = None;
      scheduled_shared_merge_node = None;
      latest_stream_id = -1;
      released = Atomic.make false;
      cross_stream_candidates = Hashtbl.create (module Tnode);
      owner_streams = Hashtbl.create (module Tnode);
      writer_stream = Hashtbl.create (module Tnode);
      reader_streams = Hashtbl.create (module Tnode);
    }

  let make_stream device runner ~stream_id =
    {
      device;
      runner;
      merge_buffer = ref None;
      scheduled_merge_node = None;
      stream_id;
      allocated_buffer = None;
      queried_work_for = Hashtbl.create (module Tnode);
    }

  let get_name stream = [%string "%{name}:%{stream.device.ordinal#Int}:%{stream.stream_id#Int}"]

  let make_context ?(ctx_arrays = Map.empty (module Tnode)) stream =
    { stream; parent = None; ctx_arrays; finalized = Atomic.make false }

  let make_child ?ctx_arrays parent =
    let ctx_arrays = Option.value ctx_arrays ~default:parent.ctx_arrays in
    { stream = parent.stream; parent = Some parent; ctx_arrays; finalized = Atomic.make false }
end

(** Parts shared by backend implementations excluding what's already in {!Backend_any_common},
    except for {!Buffer} which is duplicated for technical reasons. *)
module type Backend_impl_common = sig
  include Buffer

  val use_host_memory : bool
  (** If true, the backend will read from and write to the host memory directly whenever possible.

      [use_host_memory] can only be true on unified memory devices, like CPU and Apple Metal. *)
end

(** An interface to adding schedulers for stream-agnostic (typically CPU) backend implementations. *)
module type For_add_scheduler = sig
  include Backend_any_common

  val name : string

  include No_device_buffer_and_copying with type buffer_ptr := buffer_ptr
end

(** Lowered-level stream agnostic backend interface: implementation-facing API for CPU backends. *)
module type Lowered_no_device_backend = sig
  include Backend_impl_common
  include Backend_any_common with type buffer_ptr := buffer_ptr

  val name : string

  type procedure [@@deriving sexp_of]

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized ->
    procedure
  (** [opt_ctx_arrays], if any, already contain the arrays of the context that will result from
      linking the code. *)

  val compile_batch :
    names:string option array ->
    opt_ctx_arrays:ctx_arrays option array option ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    procedure option array
  (** [opt_ctx_arrays], if any, already contain the arrays of the contexts that will result from
      linking the code. *)

  val link_compiled :
    merge_buffer:buffer option ref ->
    runner_label:string ->
    ctx_arrays ->
    procedure ->
    Indexing.lowered_bindings * Task.t
  (** The [ctx_arrays] already contain the arrays of the resulting context. [runner_label] will be
      [get_name stream] of the stream holding the resulting [ctx_arrays]. *)

  include No_device_buffer_and_copying with type buffer_ptr := buffer_ptr
end

module type No_buffer_retrieval_or_syncing = sig
  include Backend_impl_common
  include Backend_device_common with type buffer_ptr := buffer_ptr

  val from_host : dst_ptr:buffer_ptr -> dst:context -> Ndarray.t -> unit
  (** Like {!Backend.from_host}, but without synchronization and buffer retrieval. *)

  val to_host : src_ptr:buffer_ptr -> src:context -> Ndarray.t -> unit
  (** Like {!Backend.to_host}, but without synchronization and buffer retrieval. *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst_ptr:buffer_ptr option ->
    dst:context ->
    src_ptr:buffer_ptr ->
    src:context ->
    unit
  (** Like {!Backend.device_to_device}, but without synchronization and buffer retrieval. Raises
      [Invalid_argument] if [into_merge_buffer = No] and [dst_ptr = None]. *)
end

(** A compilation-agnostic backend API -- {!Lowered_backend} instantates it, but
    {!Lowered_no_device_backend} backends are also converted to its instantations. *)
module type With_scheduler = sig
  include Backend_device_common

  val schedule_task : stream -> Task.t -> unit
end

(** Lowered-level backend interface: implementation-facing API for device-based (GPU, or CPU after
    adding a scheduler) backends. *)
module type Lowered_backend = sig
  include Backend_device_common

  include
    No_buffer_retrieval_or_syncing
      with type buffer_ptr := buffer_ptr
       and type dev := dev
       and type runner := runner
       and type event := event

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val compile : ?shared:bool -> name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    ?shared:bool ->
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    code_batch

  val link : context -> code -> ctx_arrays -> Indexing.lowered_bindings * Task.t
  (** [context] is the prior context, while [ctx_arrays] are the arrays of the resulting context.
      The results correspond to the fields {!field-Backend_intf.bindings} and
      {!field-Backend_intf.schedule} of {!Backend_intf.routine}. *)

  val link_batch :
    context ->
    code_batch ->
    ctx_arrays option array ->
    Indexing.lowered_bindings * Task.t option array
  (** [context] is the prior context, while the [ctx_arrays] are the arrays of the resulting
      contexts. Returns the schedule tasks for the procedures included in the code batch. *)
end

module Alloc_buffer_ignore_stream
    (Device_types : Device_types)
    (Backend : Alloc_buffer with type buffer_ptr = Device_types.buffer_ptr and type stream := unit) :
  Alloc_buffer with type buffer_ptr = Backend.buffer_ptr and type stream = Device_types.stream =
struct
  include Device_types

  let alloc_buffer ?old_buffer ~size_in_bytes _stream =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  let alloc_zero_init_array prec ~dims _stream = Backend.alloc_zero_init_array prec ~dims ()
  let free_buffer = Option.map Backend.free_buffer ~f:(fun memfree _stream ptr -> memfree () ptr)
end
