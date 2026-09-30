(** {1 The components for use in backend implementations}

    Implementation-facing types and components. *)

open Base
module Lazy = Utils.Lazy

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_BACKEND_IMPL=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_BACKEND_IMPL"]

open Backend_intf

(* The backend's concrete buffer-pointer type lives here, in the implementation-facing layer -- NOT
   in the shared {!Backend_intf}, which only ever speaks {!Backend_intf.buffer_loc}. It is used by
   the raw allocator and the backend-private pool tables. *)
module type Buffer = sig
  type buffer_ptr [@@deriving sexp_of]
end

module type No_device_buffer_and_copying = sig
  include Buffer

  val get_used_memory : unit -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  (** Raw slab primitives used by {!Make_slab} to back the device-level {!Backend_intf.Slab_alloc}.
      They allocate / free / zero contiguous backend buffers by byte size; pool-id bookkeeping lives
      in the shared slab wrapper. *)

  val alloc_pool_raw : size_in_bytes:int -> buffer_ptr
  val free_pool_raw : (buffer_ptr -> unit) option
  val memset_zero_raw : buffer_ptr -> offset:int -> size_in_bytes:int -> unit

  val offset_buffer : buffer_ptr -> bytes:int -> buffer_ptr
  (** Returns a handle for the slab pointer advanced by [bytes]. Used by {!Make_slab.resolve_pool}
      to turn a [{ pool_id; offset }] into the concrete pointer for a sub-region of a multi-tenant
      pool. For [bytes = 0] this must return the base unchanged. *)

  val buffer_to_buffer : dst:buffer_ptr -> src:buffer_ptr -> size_in_bytes:int -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

module No_device_buffer_and_copying () :
  No_device_buffer_and_copying with type buffer_ptr = unit Ctypes.ptr = struct
  type buffer_ptr = unit Ctypes.ptr

  let sexp_of_buffer_ptr = Ops.sexp_of_voidptr
  let used_memory = Atomic.make 0
  let get_used_memory () = Atomic.get used_memory

  let%track7_sexp alloc_pool_raw ~(size_in_bytes : int) : buffer_ptr =
    let%track7_sexp finalize (_ptr : buffer_ptr) : unit =
      ignore (Atomic.fetch_and_add used_memory ~-size_in_bytes : int)
    in
    (* Over-allocate and advance to the next [Ops.buffer_alignment] boundary: [Ctypes.allocate_n]
       (calloc) only guarantees the ABI's ~16 bytes, short of AVX/NEON vector loads (gh-ocannl-164).
       Ctypes pointer arithmetic preserves the managed root, so the zeroed allocation and GC
       lifetime semantics (derived pointers keep the buffer alive) are exactly as before. *)
    let align = Ops.buffer_alignment in
    let count = max 1 size_in_bytes + align - 1 in
    let base = Ctypes.(allocate_n int8_t ~count) in
    let pad =
      (* [align] is a power of two; masking keeps the offset non-negative even for addresses with
         the top bit set. *)
      let mask = Nativeint.of_int (align - 1) in
      let m =
        Nativeint.to_int_exn
          (Nativeint.bit_and (Ctypes.raw_address_of_ptr (Ctypes.to_voidp base)) mask)
      in
      if m = 0 then 0 else align - m
    in
    let ptr = Ctypes.(to_voidp (base +@ pad)) in
    let _ : int = Atomic.fetch_and_add used_memory size_in_bytes in
    Stdlib.Gc.finalise finalize ptr;
    ptr

  let memset_zero_raw (ptr : buffer_ptr) ~(offset : int) ~(size_in_bytes : int) : unit =
    if size_in_bytes > 0 then
      let arr = Ctypes.from_voidp Ctypes.uint8_t ptr in
      for i = offset to offset + size_in_bytes - 1 do
        Ctypes.(arr +@ i <-@ Unsigned.UInt8.zero)
      done

  let free_pool_raw = None

  let offset_buffer (base : buffer_ptr) ~(bytes : int) : buffer_ptr =
    if bytes = 0 then base else Ctypes.(to_voidp (from_voidp uint8_t base +@ bytes))

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

module Device_types_ll (Device_config : Device_config_common) = struct
  include Device_config

  type nonrec device = (dev, runner, event) device [@@deriving sexp_of]
  type nonrec context = (dev, runner, event) context [@@deriving sexp_of]
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
    [(device_id, pool_id) -> 'base] table. *)
module Make_slab (Device_types : Device_types) (Raw : No_device_buffer_and_copying) :
  Device_slab with type device = Device_types.device and type buffer_ptr = Raw.buffer_ptr = struct
  open Backend_intf

  type device = Device_types.device
  type buffer_ptr = Raw.buffer_ptr

  (* Private pool table keyed by (device_id, pool_id). The table is shared by every device of the
     backend module, and with the [Multidev] scheduler its accessors run on several domains at once:
     merge-buffer growth ([alloc_pool]) and merge-buffer resolution ([resolve_pool]) execute inside
     device tasks on worker domains, concurrently with link-time allocation and eager
     transfer-endpoint resolution on the main domain. A Base hashtable is not domain-safe -- a
     lookup racing a resize can spuriously miss an existing key -- so all accesses go through
     [pools_mutex]. *)
  let pools : (int * int, buffer_ptr) Hashtbl.Poly.t = Hashtbl.Poly.create ()
  let pools_mutex = Stdlib.Mutex.create ()
  let with_pools f = Stdlib.Mutex.protect pools_mutex f

  let alloc_pool ?mode:_ device ~pool_id ~size_in_bytes ~alignment:_ =
    let key = (device.device_id, pool_id) in
    with_pools (fun () ->
        (* The reserved merge pool is the only id replaced in place. Invalidate its complete
           ownership transaction before the fallible free/allocation sequence: growth must not
           require old+new bytes to fit simultaneously, and failure must leave no stale claim. *)
        Option.iter (Hashtbl.find pools key) ~f:(fun old ->
            if Int.equal pool_id merge_buffer_pool_id then
              invalidate_merge_slab device ~remove_slab_claim:(fun () -> Hashtbl.remove pools key)
            else Hashtbl.remove pools key;
            Option.iter Raw.free_pool_raw ~f:(fun memfree -> memfree old));
        let ptr = Raw.alloc_pool_raw ~size_in_bytes in
        if Int.equal pool_id merge_buffer_pool_id then
          commit_merge_slab device ~size_in_bytes ~install_slab_claim:(fun () ->
              Hashtbl.set pools ~key ~data:ptr)
        else Hashtbl.set pools ~key ~data:ptr)

  (* Always [Some]: even backends whose raw allocations are reclaimed by GC ([free_pool_raw = None])
     must drop the private table entry on finalization, otherwise [pools] keeps a strong reference
     to every tnode buffer for the lifetime of the backend module and the GC finalizer never runs.
     Removing the entry releases that reference (and eagerly frees via the raw deallocator if
     any). *)
  let free_pool =
    Some
      (fun device ~pool_id ->
        let key = (device.device_id, pool_id) in
        with_pools (fun () ->
            Option.iter (Hashtbl.find pools key) ~f:(fun ptr ->
                Option.iter Raw.free_pool_raw ~f:(fun memfree -> memfree ptr));
            Hashtbl.remove pools key))

  let memset_zero device ~pool_id ~offset ~size_in_bytes =
    let ptr = with_pools (fun () -> Hashtbl.find_exn pools (device.device_id, pool_id)) in
    Raw.memset_zero_raw ptr ~offset ~size_in_bytes

  let resolve_pool device { pool_id; offset } =
    (* Pooled policy: many tnodes share a pool at distinct byte offsets. Resolve to the slab base
       and advance by [offset] via the backend's raw pointer arithmetic. *)
    let base = with_pools (fun () -> Hashtbl.find_exn pools (device.device_id, pool_id)) in
    Raw.offset_buffer base ~bytes:offset
end

let next_global_device_id : Utils.atomic_int = Atomic.make 0

module Device
    (Device_types : Device_types)
    (Slab : Device_slab with type device := Device_types.device) =
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
  let classify_failure _phase _exn = None

  let make_context ?(ctx_buffers = Map.empty (module Tnode)) ?optimize_ctx device =
    let optimize_ctx = Option.value_or_thunk optimize_ctx ~default:Low_level.empty_optimize_ctx in
    Alloc_census.count_context_created ();
    {
      device;
      parent = None;
      ctx_buffers;
      finalized = Atomic.make false;
      released_pool_ids = Set.empty (module Int);
      upload_arenas = fresh_upload_arenas ();
      optimize_ctx;
      merge_buffer_node = None;
    }

  let make_child ?ctx_buffers ?optimize_ctx ?merge_buffer_node parent =
    let ctx_buffers = Option.value ctx_buffers ~default:parent.ctx_buffers in
    let optimize_ctx = Option.value optimize_ctx ~default:parent.optimize_ctx in
    let merge_buffer_node = Option.value merge_buffer_node ~default:parent.merge_buffer_node in
    Alloc_census.count_context_created ();
    {
      device = parent.device;
      parent = Some parent;
      ctx_buffers;
      finalized = Atomic.make false;
      released_pool_ids = Set.empty (module Int);
      upload_arenas = fresh_upload_arenas ();
      optimize_ctx;
      merge_buffer_node;
    }
end

(** An interface to adding schedulers for stream-agnostic (typically CPU) backend implementations.
*)
module type For_add_scheduler = sig
  val name : string
  val codegen_capabilities : unit -> codegen_capabilities

  include No_device_buffer_and_copying
end

(** Lowered-level stream agnostic backend interface: implementation-facing API for CPU backends. *)
module type Lowered_no_device_backend = sig
  include Buffer

  val name : string
  val codegen_capabilities : unit -> codegen_capabilities

  type procedure [@@deriving sexp_of]

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> procedure

  val compile_batch :
    names:string array -> Indexing.unit_bindings -> Low_level.optimized array -> procedure array
  (** Compiles the given procedures -- the segment kernels of one fissioned routine -- as a single
      compilation unit: one generated source and one C-compiler invocation, with the resulting
      (dyn-loaded) library shared by every returned procedure. *)

  val link_compiled :
    ?lowered_bindings:Indexing.lowered_bindings ->
    merge_buffer:Backend_intf.buffer_loc option ref ->
    resolve:(Backend_intf.buffer_loc -> buffer_ptr) ->
    runner_label:string ->
    Backend_intf.ctx_buffers ->
    procedure ->
    Indexing.lowered_bindings * Task.t
  (** [resolve] is the device's backend-private [buffer_loc -> base] lookup, supplied at the backend
      boundary so that the {e shared} layer never handles a raw pointer: this function resolves both
      the context's [ctx_buffers] (eagerly, at link time) and the lazily-set [merge_buffer] (at
      execution time). [runner_label] is [get_name device] of the device holding the buffers.
      [lowered_bindings], when given, supplies the static-index refs to bind (looked up by symbol)
      instead of freshly minted ones, and is returned as-is: procedures linked as a batch — in
      particular the segment kernels of one fissioned routine — must share their binding refs, so
      setting a static index through the routine's bindings reaches every kernel. *)

  include No_device_buffer_and_copying with type buffer_ptr := buffer_ptr
end

(** The transfer/sync seam the shared {!Backends} layer consumes. It speaks
    {!Backend_intf.buffer_loc} only -- the concrete backend pointer never crosses this boundary;
    each backend resolves [buffer_loc -> base] internally. *)
module type No_buffer_retrieval_or_syncing = sig
  include Buffer
  include Backend_device_common

  val from_host : dst:context -> dst_loc:Backend_intf.buffer_loc -> Ndarray.t -> unit
  (** Like {!Backend_intf.Backend.from_host}, but without synchronization and buffer retrieval; the
      backend resolves [dst_loc] against [dst.device]'s private pool table. *)

  val to_host : src:context -> src_loc:Backend_intf.buffer_loc -> Ndarray.t -> unit
  (** Like {!Backend_intf.Backend.to_host}, but without synchronization events and buffer retrieval;
      the backend resolves [src_loc] against [src.device]'s private pool table. *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst_loc:Backend_intf.buffer_loc option ->
    dst:context ->
    src_loc:Backend_intf.buffer_loc ->
    src:context ->
    unit
  (** Like {!Backend_intf.Backend.device_to_device}, but without synchronization events and buffer
      retrieval; the backend resolves the locations internally. Raises [Invalid_argument] if
      [into_merge_buffer = No] and [dst_loc = None]. *)
end

(** An intermediate stage for converting {!Lowered_no_device_backend} backends into
    {!Lowered_backend}. This impl-facing stage may carry the backend-private [resolve_pool] (its
    base type does not escape to {!Backend_intf}). *)
module type With_scheduler = sig
  include Backend_device_common
  include Buffer

  val resolve_pool : device -> Backend_intf.buffer_loc -> buffer_ptr
  (** Backend-private [buffer_loc -> base] resolution, used by the backend's own transfer/link
      implementations. Not part of {!Backend_intf}. *)

  val schedule_task : device -> Task.t -> unit
end

(** Lowered-level backend interface: implementation-facing API for device-based (GPU, or CPU after
    adding a scheduler) backends based on the {!Low_level} IR. *)
module type Lowered_backend = sig
  include Backend_device_common

  include
    No_buffer_retrieval_or_syncing
      with type dev := dev
       and type runner := runner
       and type event := event

  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    names:string array -> Indexing.unit_bindings -> Low_level.optimized array -> code_batch
  (** Compiles the given procedures -- the segment kernels of one fissioned routine -- as a single
      compilation unit (one generated source, one backend-compiler invocation, one module to load at
      link time). Ideally does not affect execution relative to separate [compile]s, but there can
      be backend-specific differences. *)

  val link : context -> code -> ctx_buffers -> Indexing.lowered_bindings * Task.t
  (** [context] is the prior context, while [ctx_buffers] are the locations of the resulting
      context. The results correspond to the fields {!field:Backend_intf.bindings} and
      {!field:Backend_intf.schedule} of {!Backend_intf.routine}. *)

  val link_batch : context -> code_batch -> ctx_buffers -> Indexing.lowered_bindings * Task.t array
  (** [context] is the prior context, while [ctx_buffers] are the locations of the resulting context
      -- ONE buffers delta shared by the whole batch, since the batch is the segment kernels of one
      fissioned routine. Returns the schedule tasks of the batch's procedures, in order, sharing one
      set of static-index refs (so setting a binding through the routine reaches every kernel). *)

  val sequence_segments :
    context ->
    name:string ->
    bindings:Indexing.lowered_bindings ->
    uses_merge_buffer:bool ->
    Task.t list ->
    Task.t option
  (** When the backend can run an ordered batch of same-stream kernel tasks — the segment schedules
      of one fissioned routine, all from this backend's [link_batch] on [context]'s device — with
      device-side ordering cheaper than per-boundary events, returns the combined task. E.g. the
      Metal backend encodes every segment's dispatch into one command buffer whose serial compute
      pass executes them in encoding order, replacing two event command buffers per boundary; the
      CUDA and HIP backends stream-capture the launch sequence into a graph replayed as one API call
      per step. [bindings] are the routine's static-index refs and [uses_merge_buffer] says whether
      the routine reads the device's merge buffer: together they cover every launch-time-varying
      kernel argument, so a backend that bakes arguments (graph capture) can key its captures on
      their current values. [None] falls back to the generic event chain of [Raise_backend.link]. *)
end
