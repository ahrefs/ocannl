(** {1 The components for use in backend implementations}

    Implementation-facing types and components. *)

open Base
module Lazy = Utils.Lazy

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_BACKEND_IMPL=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_BACKEND_IMPL"]

open Backend_intf

module type No_device_buffer_and_copying = sig
  include Buffer

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option

  val get_used_memory : unit -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  (** Raw slab primitives used by {!Make_slab} to back the device-level {!Backend_intf.Slab_alloc}.
      They allocate / free / zero contiguous backend buffers by byte size; pool-id bookkeeping lives
      in the shared slab wrapper. *)

  val alloc_pool_raw : size_in_bytes:int -> buffer_ptr
  val free_pool_raw : (buffer_ptr -> unit) option
  val memset_zero_raw : buffer_ptr -> offset:int -> size_in_bytes:int -> unit
  val buffer_to_buffer : dst:buffer_ptr -> src:buffer_ptr -> size_in_bytes:int -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

module No_device_buffer_and_copying () :
  No_device_buffer_and_copying with type buffer_ptr = unit Ctypes.ptr = struct
  type buffer_ptr = unit Ctypes.ptr

  let use_host_memory = Some (fun ~size_in_bytes:_ ptr -> ptr)
  let sexp_of_buffer_ptr = Ops.sexp_of_voidptr

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)

  let used_memory = Atomic.make 0
  let get_used_memory () = Atomic.get used_memory

  let%track7_sexp alloc_pool_raw ~(size_in_bytes : int) : buffer_ptr =
    let%track7_sexp finalize (_ptr : buffer_ptr) : unit =
      ignore (Atomic.fetch_and_add used_memory ~-size_in_bytes : int)
    in
    let ptr = Ctypes.(to_voidp @@ allocate_n int8_t ~count:(max 1 size_in_bytes)) in
    let _ : int = Atomic.fetch_and_add used_memory size_in_bytes in
    Stdlib.Gc.finalise finalize ptr;
    ptr

  let memset_zero_raw (ptr : buffer_ptr) ~(offset : int) ~(size_in_bytes : int) : unit =
    if size_in_bytes > 0 then (
      let arr = Ctypes.from_voidp Ctypes.uint8_t ptr in
      for i = offset to offset + size_in_bytes - 1 do
        Ctypes.(arr +@ i <-@ Unsigned.UInt8.zero)
      done)

  let free_pool_raw = None

  type void_buffer_ptr = (Stdlib.Obj.t option, unit Ctypes_static.typ) Ctypes_ptr.Fat.t

  let sexp_of_void_buffer_ptr (p : void_buffer_ptr) =
    Sexp.Atom (Ctypes_value_printing_stubs.string_of_pointer p)

  let () = ignore sexp_of_void_buffer_ptr

  let%track7_sexp memcpy ~(dst : void_buffer_ptr) ~(src : void_buffer_ptr) ~(size_in_bytes : int) :
      unit =
    if Ctypes_ptr.Fat.compare dst src <> 0 then
      Ctypes_memory_stubs.memcpy ~dst ~src ~size:size_in_bytes

  let buffer_to_buffer ~dst:Ctypes_static.(CPointer dst) ~src:Ctypes_static.(CPointer src)
      ~size_in_bytes =
    memcpy ~dst ~src ~size_in_bytes

  let host_to_buffer src ~dst:Ctypes_static.(CPointer dst) =
    memcpy ~dst ~src:(Ndarray.get_fatptr_not_managed src) ~size_in_bytes:(Ndarray.size_in_bytes src)

  let buffer_to_host dst ~src:Ctypes_static.(CPointer src) =
    memcpy ~dst:(Ndarray.get_fatptr_not_managed dst) ~src ~size_in_bytes:(Ndarray.size_in_bytes dst)
end

module Device_types (Device_config : Device_config) = struct
  include Device_config

  type nonrec device = (dev, runner, event) device [@@deriving sexp_of]
  type nonrec context = (dev, runner, event, optimize_ctx) context [@@deriving sexp_of]
end

module Device_types_ll (Device_config : Device_config_common) = struct
  include Device_config

  type optimize_ctx = Low_level.optimize_ctx [@@deriving sexp_of]

  let empty_optimize_ctx () = { Low_level.computations = Hashtbl.create (module Tnode) }

  type nonrec device = (dev, runner, event) device [@@deriving sexp_of]
  type nonrec context = (dev, runner, event, Low_level.optimize_ctx) context [@@deriving sexp_of]
end

(** The device-level slab interface a {!Device} functor consumes: the {!Backend_intf.Slab_alloc}
    primitives plus the [resolve_pool] address resolution. *)
module type Device_slab = sig
  type device

  include Backend_intf.Slab_alloc with type device := device

  type buffer_ptr

  val resolve_pool : device -> Backend_intf.buffer_loc -> buffer_ptr
end

(** Backs the device-level slab interface with a backend's raw byte-buffer primitives and a private
    [(device_id, pool_id) -> 'base] table. Replaces the old [Alloc_buffer_ignore_stream]. *)
module Make_slab
    (Device_types : Device_types)
    (Raw : No_device_buffer_and_copying with type buffer_ptr = Device_types.buffer_ptr) :
  Device_slab
    with type device = Device_types.device
     and type buffer_ptr = Device_types.buffer_ptr = struct
  open Backend_intf

  type device = Device_types.device
  type buffer_ptr = Device_types.buffer_ptr

  (* Private pool table keyed by (device_id, pool_id). *)
  let pools : (int * int, buffer_ptr) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode:_ device ~pool_id ~size_in_bytes ~alignment:_ =
    let ptr = Raw.alloc_pool_raw ~size_in_bytes in
    Hashtbl.set pools ~key:(device.device_id, pool_id) ~data:ptr

  let free_pool =
    Option.map Raw.free_pool_raw ~f:(fun memfree device ~pool_id ->
        let key = (device.device_id, pool_id) in
        Option.iter (Hashtbl.find pools key) ~f:memfree;
        Hashtbl.remove pools key)

  let memset_zero device ~pool_id ~offset ~size_in_bytes =
    let ptr = Hashtbl.find_exn pools (device.device_id, pool_id) in
    Raw.memset_zero_raw ptr ~offset ~size_in_bytes

  let resolve_pool device { pool_id; offset } =
    (* Phase-1 policy: one pool per tnode at offset 0. Aliasing (offset > 0) is future work and would
       add backend-specific pointer arithmetic here. *)
    assert (offset = 0);
    Hashtbl.find_exn pools (device.device_id, pool_id)
end

let next_global_device_id : Utils.atomic_int = Atomic.make 0

module Device
    (Device_types : Device_types)
    (Slab :
      Device_slab
        with type buffer_ptr := Device_types.buffer_ptr
         and type device := Device_types.device) =
struct
  include Device_types
  include Slab

  let make_device dev runner ~ordinal =
    let device_id = Atomic.fetch_and_add next_global_device_id 1 in
    {
      dev;
      ordinal;
      device_id;
      runner;
      merge_buffer = ref None;
      merge_buffer_capacity = 0;
      updating_for = Hashtbl.create (module Tnode);
      updating_for_merge_buffer = None;
      constant_buffer_cache = Hashtbl.create (module Tnode);
      next_pool_id = merge_buffer_pool_id + 1;
    }

  let get_name device = [%string "%{name}:%{device.ordinal#Int}:%{device.device_id#Int}"]

  let make_context ?(ctx_buffers = Map.empty (module Tnode)) ?optimize_ctx device =
    let optimize_ctx = Option.value_or_thunk optimize_ctx ~default:empty_optimize_ctx in
    {
      device;
      parent = None;
      ctx_buffers;
      finalized = Atomic.make false;
      optimize_ctx;
      merge_buffer_node = None;
    }

  let make_child ?ctx_buffers ?optimize_ctx ?merge_buffer_node parent =
    let ctx_buffers = Option.value ctx_buffers ~default:parent.ctx_buffers in
    let optimize_ctx = Option.value optimize_ctx ~default:parent.optimize_ctx in
    let merge_buffer_node =
      Option.value merge_buffer_node ~default:parent.merge_buffer_node
    in
    {
      device = parent.device;
      parent = Some parent;
      ctx_buffers;
      finalized = Atomic.make false;
      optimize_ctx;
      merge_buffer_node;
    }
end

(** Parts shared by backend implementations. *)
module type Backend_impl_common = sig
  include Backend_intf.Buffer

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option
  (** If not [None], the backend will read from and write to the host memory directly whenever
      reasonable. [size_in_bytes] is the size of the memory allocated on the host.

      [use_host_memory] can only be [Some] on unified memory devices, like CPU and Apple Metal. *)
end

(** An interface to adding schedulers for stream-agnostic (typically CPU) backend implementations.
*)
module type For_add_scheduler = sig
  val name : string

  include No_device_buffer_and_copying
end

(** Lowered-level stream agnostic backend interface: implementation-facing API for CPU backends. *)
module type Lowered_no_device_backend = sig
  include Backend_impl_common

  val name : string

  type procedure [@@deriving sexp_of]

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> procedure

  val compile_batch :
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    procedure option array

  val link_compiled :
    merge_buffer:Backend_intf.buffer_loc option ref ->
    resolve:(Backend_intf.buffer_loc -> buffer_ptr) ->
    runner_label:string ->
    buffer_ptr Map.M(Tnode).t ->
    procedure ->
    Indexing.lowered_bindings * Task.t
  (** The [buffer_ptr] map already contains the resolved pointers of the resulting context (the
      shared layer resolves [ctx_buffers] before calling). [resolve] resolves the (lazily set)
      [merge_buffer] location at execution time. [runner_label] is [get_name device] of the device
      holding the resulting buffers. *)

  include No_device_buffer_and_copying with type buffer_ptr := buffer_ptr
end

module type No_buffer_retrieval_or_syncing = sig
  include Backend_impl_common
  include Backend_device_common with type buffer_ptr := buffer_ptr

  val from_host : dst_ptr:buffer_ptr -> dst:context -> Ndarray.t -> unit
  (** Like {!Backend_intf.Backend.from_host}, but without synchronization and buffer retrieval. *)

  val to_host : src_ptr:buffer_ptr -> src:context -> Ndarray.t -> unit
  (** Like {!Backend_intf.Backend.to_host}, but without synchronization events and buffer retrieval.
  *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst_ptr:buffer_ptr option ->
    dst:context ->
    src_ptr:buffer_ptr ->
    src:context ->
    unit
  (** Like {!Backend_intf.Backend.device_to_device}, but without synchronization events and buffer
      retrieval. Raises [Invalid_argument] if [into_merge_buffer = No] and [dst_ptr = None]. *)
end

(** An intermediate stage for converting {!Lowered_no_device_backend} backends into
    {!Lowered_backend}. It could potentially be used for assignments-level backends too. *)
module type With_scheduler = sig
  include Backend_device_common

  val schedule_task : device -> Task.t -> unit
end

(** Lowered-level backend interface: implementation-facing API for device-based (GPU, or CPU after
    adding a scheduler) backends based on the {!Low_level} IR. *)
module type Lowered_backend = sig
  include Backend_device_common with type optimize_ctx := Low_level.optimize_ctx

  include
    No_buffer_retrieval_or_syncing
      with type buffer_ptr := buffer_ptr
       and type dev := dev
       and type runner := runner
       and type event := event
       and type optimize_ctx := Low_level.optimize_ctx

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    code_batch

  val link : context -> code -> ctx_buffers -> Indexing.lowered_bindings * Task.t
  (** [context] is the prior context, while [ctx_buffers] are the locations of the resulting context.
      The results correspond to the fields {!field:Backend_intf.bindings} and
      {!field:Backend_intf.schedule} of {!Backend_intf.routine}. *)

  val link_batch :
    context ->
    code_batch ->
    ctx_buffers option array ->
    Indexing.lowered_bindings * Task.t option array
  (** [context] is the prior context, while the [ctx_buffers] are the locations of the resulting
      contexts. Returns the schedule tasks for the procedures included in the code batch. *)
end
