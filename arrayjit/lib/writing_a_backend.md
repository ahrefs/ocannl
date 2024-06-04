# The Anatomy of an OCANNL Backend

<!-- TOC -->

- [The Anatomy of an OCANNL Backend](#the-anatomy-of-an-ocannl-backend)
  - [Design around compiling and running code, backend interfaces](#design-around-compiling-and-running-code-backend-interfaces)
    - [Shared relocatable compilation, batch compilation](#shared-relocatable-compilation-batch-compilation)
  - [Tensor nodes, arrays, memory properties](#tensor-nodes-arrays-memory-properties)
  - [Typical details of a backend implementation](#typical-details-of-a-backend-implementation)
    - [Conditionally emitting the tracing debugger code](#conditionally-emitting-the-tracing-debugger-code)
  - [Synchronization and data transfers](#synchronization-and-data-transfers)
    - [Data transfers](#data-transfers)
    - [Synchronization](#synchronization)

<!-- /TOC -->

## Design around compiling and running code, backend interfaces

Currently, OCANNL integrates new backends via code in [backends.ml](backends.ml). `Backends` introduces the context-specific `routine` type, and has a helper `lower_assignments` that wraps `Assignments.lower` and `Low_level.optimize_proc`. The interface to a `Backend` has `compile` functions, to allow some backends handle assignments directly, instead of using the optimized C-like representation `Low_level.t`.

```ocaml
type lowered_bindings = (static_symbol, int ref) List.Assoc.t  (* in indexing.ml *)

type task = Work of ((module Debug_runtime) -> unit -> unit)  (* in tnode.ml *)

type 'context routine = {
  context : 'context;
  schedule : task;
  bindings : lowered_bindings;
  name : string;
}

module type No_device_backend = sig
  type code
  type context
  type nonrec routine = context routine
  ...
end
```

Backends need to provide the `code` (for compilation result) and `context` types. `code` is some intermediate state between assignments and `context routine`. A backend may postpone doing anything specific until linking, e.g. `code = Low_level.optimized`, or linking may be a no-op, effectively `code = routine`, usually it will fall somewhere in between and depend on whether `~shared:true` is passed. For simple situations like CPU backends, `Backends` has a helper functor:

```ocaml
module Simple_no_device_backend (Backend : Simple_backend) : No_device_backend = struct
  type code =
    | Postponed of { lowered : Low_level.optimized; bindings : Indexing.unit_bindings; name : string }
    | Compiled of Backend.procedure
  [@@deriving sexp_of]

  include Backend

  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?(shared = false) ?name bindings asgns =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  ...
end
```

where `Simple_backend` implements a `No_device_backend` functionality, but only needs to deal with `Low_level.optimized` and its compilation result type `procedure`.

`No_device_backend`s do not themselves deal with the device abstraction. There's the functor `Multicore_backend (Backend : No_device_backend) : Backend` that assigns a device to a domain, and manages the given `No_device_backend` on the domain-based devices. Running `schedule` on a `No_device_backend` _should block_ (till execution finishes), but it _should not block_ for a proper `Backend` -- it should just put the work on the device's queue.

```ocaml
module type Backend = sig
  include No_device_backend
  ...
  type physical_device
  type device
  val init : device -> context
  val await : device -> unit
  val is_idle : device -> bool
  val get_device : ordinal:int -> physical_device
  val get_physical_device : device -> physical_device
  val new_virtual_device : physical_device -> device
  ...
end
```

`Backend.await` synchronizes the device -- waits for all work on the device to finish -- the device becomes `is_idle`.

When devices natively implement a lightweight threads mechanism, as CUDA does via _streams_, the lightweight threads are exposed via `new_virtual_device` generating a fresh thread. Otherwise, `physical_device = device` and the functions `new_virtual_device` and `get_physical_device` are identities.

### Shared (relocatable) compilation, batch compilation

Shared (relocatable) compilation, with `~shared:true`, improves compilation efficiency, because code can be compiled once for use on multiple devices (in multiple contexts). It also improves debugging convenience, by generating fewer debugging artifacts. A potential downside is slightly less efficient computations.

Batched compilation has similar benefits, especially in producing fewer debugging artifacts. The compilation might also be slightly more efficient since the compiler needs to be invoked fewer times. While `~shared:true` compilation saves (more) total work by compiling _one routine for many devices_, batched compilation and linking saves a bit of work by compiling _many routines for one device_ at once.

## Tensor nodes, arrays, memory properties

OCANNL classifies tensor nodes according to their memory properties:

 ```ocaml
type memory_type =
  | Constant  (** The tensor node does not change after initialization. *)
  | Nonconstant  (** One of: [Changed_on_devices], [Volatile]. *)
  | Changed_on_devices  (** The tensor node will only change on host via a [to_host] call. *)
  | Volatile  (** The tensor node will only change on any device via a [from_host] or [merge] call. *)

type memory_mode =
  | Effectively_constant  (** Either [Hosted Constant], or a subset of [Virtual]. *)
  | Virtual  (** The tensor node's computations are inlined on a per-scalar basis. *)
  | Never_virtual  (** One of: [Local], [On_device], [Hosted]. *)
  | Local
      (** The full tensor node is cached for the duration of a computation but not persisted across calls to
          compiled functions. It is not available for merging across devices. *)
  | Device_only  (** One of: [Local], [On_device]. *)
  | On_device
      (** The tensor node is stored on the devices that compute with it and persisted across function calls.
          It is available for merging across devices (for devices that support merging / P2P), but not
          (directly) for visualization or storing to disk. *)
  | Materialized  (** One of: [On_device], [Hosted]. *)
  | Hosted of memory_type
      (** The tensor node is stored in a globally addressable memory, in addition to on devices where it is
          computed with (or as part of one of them, if "hosting on device", or only on the host and not on
          devices, for some backends). It is available for all operations, and visible to OCaml programs as an
          {!Ndarray} (the optional [array] of {!t}). *)
 ```

 `Tnode.update_memory_mode` verifies consistency of the updates of these modes. Currently, these properties are only either set explicitly (directly or indirectly) by the user, or determined by the `Low_level` analysis and optimization process. Since backends can have full control of the optimizations, in the future determining the memory mode can also be backend-specific.

Backends also have their specific memory classes for how arrays are stored. These are needed because, for example, `Hosted` does not precisely specify the places where the memory of an array is allocated, and a backend itself can have multiple ways of string an array on devices. And the distinction between `On_device` and `Hosted` may not be relevant for how arrays are manipulated on the backend (except for the behavior of `to_host`, `from_host`). Moreover, the memory classes used by a backend can evolve as the backend becomes more sophisticated. CPU backends use:

```ocaml
type mem_properties =
  | Local_only  (** The array is only needed for a local computation, is allocated on the stack. *)
  | From_context  (** The array has a copy allocated per-cpu-device, may or may not exist on the host. *)
  | Constant_from_host  (** The array is read directly from the host. *)
```

and our initial, naive CUDA backend uses:

```ocaml
type mem_properties =
  | Local_only
      (** The array is only needed for a single computation and is allocated locally (or spilled). *)
  | Global  (** Could not perform optimizations: the array is computed directly in the global memory. *)
```

Hopefully soon we will support arrays shared across a thread block, then the above will include `Shared`.

You might notice that `Simple_backend` requires:

```ocaml
module type Simple_backend = sig
  type context
  type procedure
  type ctx_arrays

  val ctx_arrays : context -> ctx_arrays

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized ->
    procedure

  ...

  val link_compiled :
    context -> procedure -> context * Indexing.lowered_bindings * Tnode.task * string

  ...
end
```

Contexts track (or store) the on-device arrays corresponding to tensor nodes. Contexts form a hierarchy: linking takes a parent context and outputs a child context. Related contexts that use a tensor node must use the same on-device array for the tensor node. If two unrelated contexts are on the same device, i.e. have a common ancestor, and use the same tensor node, the behavior is undefined. For CPU backends, the arrays might be stored as:

```ocaml
type ctx_arrays = Ndarray.t Map.M(Tn).t
```

For a CUDA backend, the arrays might be tracked as:

```ocaml
  global_arrays : Cudajit.deviceptr Map.M(Tn).t;
```

## Typical details of a backend implementation

A backend needs to maintain information about tensor nodes, in a datatype conventionally called `tn_info`, that might look like this:

```ocaml
type tn_info = {
  tn : Tn.t;  (** The original array. *)
  ptr : ...(* Some way to refer to the array from the compiled code. *);
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  num_typ : ...;
      (** The type of the stored values:
          [short] or [half] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
  zero_initialized : bool;
  ...
}
```

`tn_info` values are typically called `node` for readability. The `tn_info`s are stored inside a compilation state datatype typically called `info` or `info_nodes`. During the compilation process, the new context is not available, and even the old context cannot be available if the backend supports shared compilation. A backend may for simplicity not suport shared compilation, i.e. ignore `~shared:true` and postpone compilation to the linking phase. Currently, the CUDA backend does the opposite, it ignores `~shared:false` and always generates relocatable kernels. This does not require any extra compilation flag, because the kernels refer to context (i.e. global) arrays via parameters. We face two cases:

- Non-trivial `~shared:true`: `tn_info`s are by necessity generated from scratch (either during compilation inside a `get_array`, or at once via `prepare_arrays` as in the `gccjit` backend). If this is the only mode the backend supports, they don't need to be stored.
- Non-trivial `~shared:false`: `tn_info`s must be propagated via the context, because to benefit from not sharing, `tn_info` must include context-specific information (typically a memory pointer to the on-device array).

The `gccjit` backend needs to `prepare_arrays` upfront, because it needs to know the list of parameters of the compiled function before it starts the compilation. This forces the `gccjit` backend to postpone creating the array pointers (via lazy initializers), not because of the context array pointers, but because of the local array pointers (on the function stack) which require knowing the function.

Conventionally, the compilation implementation is split into three functions / layers:

- `compile_main` does the bulk of translating a `Low_level.t` into the backend-specific code.
- `compile_proc` populates the parameters in the function header, fills-in the function's initialization section that sets up local arrays, clears these arrays (whether local or global) that need to be zero-initialized or reset to zero, appends the `compile_main` code.
- `compile`, resp. `compile_batch`, compiles the function, resp. functions, into an executable object/file or assembly object/file.
  - On same-machine CPU backends, these functions also dynamically load (if applicable) the code (since there's shared program memory for all cores) -- `compile_batch` includes the same resulting (dyn-loaded) object in the code output for all functions.
  - On GPU-like backends, we cannot load the code at compile time. For example, the CUDA driver API function `cuModuleLoadDataEx` loads the module into _the current context_, which is device-specific, so it must be called from within `link` or `link_batch`.
    - Within `link` and `link_batch`, the `cuda` backend first calls `Cudajit.ctx_set_current` (if needed) and only then `Cudajit.module_load_data_ex`.
    - GPU-like backends necessitate distinguishing between `link` and `link_batch`, to prevent the same code from being loaded as multiple modules.

### Conditionally emitting the tracing debugger code

Backends should support logging some of the computations when `Utils.settings.debug_log_from_routines` is set. Obviously, setting this debug information only makes sense for tiny models / computations, otherwise the log files will explode. For GPU backends, cleanly logging from more than one thread per device would add too much complexity, so we restrict logging to `blockIdx = 0, threadIdx = 0`.

We output a log line only for comments and array assignments (corresponding to non-virtual node computations), but we log the computed expression structure: the indices, array values, and inline computation values. For simplicity and conciseness, we don't log the structure of inlined computations. Comments determine the nesting of the `ppx_minidebug` entries: when lowering a `Block_comment`, `Assignments.to_low_level` outputs a `Comment "end"` at the end of a block. Comments are prefixed with `COMMENT:`. For assignments, we also log the debug information stored in the `Set` construct -- it's the computed expression translated into the `%cd` syntax. For processing, the debug information is prefixed by `#` and has endlines replaced by `$`. These structural prefixes / infixes are parsed out by `Utils.log_trace_tree`.

Since the CUDA backend needs to capture the standard output, it additionally prefixes each log line with a kernel run ID. When postprocessing the logs, each run extracts its own log lines. Simultaneous logging from multiple CUDA devices should still be clean -- without interleaving lines -- because the driver is supposed to dump the logs to standard output at device synchronization points.

## Synchronization and data transfers

Currently, OCANNL expects backends to implement a FIFO queue scheduling mechanism. The scheduling does not express dependencies between tensors. Only the main domain is allowed to interact with devices (queues with single producer -- host, single consumer -- virtual device).

Since this is significantly simpler than what other frameworks do, it might evolve in the future. (In particular, scheduling in `tinygrad` expresses tensor graph dependencies with arbitrary queueing.) A natural next step would be to add "acknowledge" events that indirectly keep track of (and signal) which tasks a device has already executed.

Besides routines, calling `from_host`, `to_host`, `device_to_device` from a backend puts the corresponding tasks on the device's queue. Implementations of `No_device_backend` and `Simple_backend` (i.e. CPU backends) should run the tasks by executing them directly.

### Data transfers

OCANNL supports asynchronous data transfers by embedding them in the scheduling mechanism.

```ocaml
module type No_device_backend = sig
  ...
  val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
  (** If the array is both hosted and in-context, schedules a copy from host to context, otherwise does
      nothing. NOTE: when run for a device, it's the caller's responsibility to synchronize the device
      before the host's data is overwritten. *)

  val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> unit
  (** If the array is both hosted and in-context, schedules a copy from context to host, otherwise does
      nothing. NOTE: when run for a device, it's the caller's responsibility to synchronize the device
      before the host's data is read. *)

  val device_to_device :
    ?rt:(module Minidebug_runtime.Debug_runtime) ->
    Tnode.t ->
    into_merge_buffer:bool ->
    dst:context ->
    src:context ->
    unit
  (** If the array is present in the [src] context, and in the [dst] context in case
      [~into_merge_buffer:false], schedules a copy of the tensor node from the device of [src] to the device
      of [dst]; otherwise, if [~into_merge_buffer:true], unsets the merge buffer source on [dst]. If
      [~into_merge_buffer:false], copies into the given node on [dst]; if [~into_merge_buffer:true], sets on
      [dst] the merge buffer source to the given node and handles the transfer using it. NOTE: when run for a
      device, it's the caller's responsibility to synchronize the [src] device, if needed, {i before} calling
      [device_to_device], and the [dst] device {i afterward}, according to {!physical_merge_buffers}, before
      any computations on the [src] device overwrite the node. *)

  val physical_merge_buffers : bool
  (** If [physical_merge_buffers = true], the [src] node data is not needed after the task scheduled by
      [device_to_device] finishes. Otherwise, the [src] node data may be needed till after the tasks computing
      with the merge buffer finish. *)
  ...
end
```

OCANNL provides explicit _merge buffers_ for performing tensor node updates, where different versions of a tensor node from two devices feature in the same computation. The `%cd` syntax for using merge buffers is via applying `merge_buffer` pseudo-function. For example, the code for merging gradients might be: `[%cd p.grad =+ merge_buffer p.grad]`. Currently, OCANNL supports only one merge buffer per device. We keep track of the specific tensor node that occupies this buffer, so we detect potential mismatches when a code's task is run. Backends where a device cannot directly access another device's memory (as far as the backend's implementer is concerned) use physical arrays to back merge buffers. There's only one buffer array per device and it is resized (grown) if needed to fit a node's array. CPU backends can simply record the other device's array pointer in the merge buffer. In the future we might implement intermediate solutions based on memory streaming.

### Synchronization

The `Utils` module provides a thread-safe (thanks to atomic flags) `waiter` mechanism for suspending and resuming threads. Currently, `waiter`s only support sequential `await` events (as needed by the single-producer single-consumer queues). This can be easily generalized to allow concurrent `await` events. `await` could take an identifier of the waiting thread, and `release_if_waiting` could return an optional identifier of the thread that got resumed.

The `Backends.Multicore_backend` functor implements scheduling with lock-free single-producer single-consumer queues. Thread safety is ensured, because each device:

- uses two (thread-safe) `waiter`s, one for each of the communicating threads (host and device), so that `await` resp. `release_if_waiting` only happen from their respective threads,
- uses two position pointers into the work queue, each thread (host resp. device) only modifies its position pointer,
- each `await` is delineated (with an up to 5 second periodic check) to ensure that the host doesn't wait for an inactive device, and that waiting ends in a well-defined state: host's waiting to synchronize a device ends when the device started waiting; plus some defensive checks that should never be actually needed.
