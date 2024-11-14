# The Anatomy of an OCANNL Backend

<!-- TOC -->

- [The Anatomy of an OCANNL Backend](#the-anatomy-of-an-ocannl-backend)
  - [Design around compiling and running code, backend interfaces](#design-around-compiling-and-running-code-backend-interfaces)
    - [Shared relocatable compilation, batch compilation](#shared-relocatable-compilation-batch-compilation)
  - [Tensor nodes, arrays, memory properties](#tensor-nodes-arrays-memory-properties)
  - [Typical details of a backend implementation](#typical-details-of-a-backend-implementation)
    - [Conditionally emitting the tracing debugger code](#conditionally-emitting-the-tracing-debugger-code)
      - [Tracing via stdout](#tracing-via-stdout)
  - [Synchronization and data transfers](#synchronization-and-data-transfers)
    - [Data transfers](#data-transfers)

<!-- /TOC -->

## Design around compiling and running code, backend interfaces

The modules and files of `arrayjit` can loosely be divided into three parts.

- "Utility": generic code providing support across the project, both for `arrayjit` and `neural_nets_lib`.
  - The modules `Utils`, `Rand` and `Ppx_helper`.
- "Frontend": the parts that are user-visible, concrete, and shared across the project.
  - `Task`: a wrapper for work execution (`unit -> unit`).
  - `Ops`: numeric precision specification types, primitive numerical operations.
  - `Ndarray`: a wrapper around bigarrays hiding their numeric precision, with accessing and `PrintBox`-based rendering.  
  - `Tnode`: the _tensor node_ type: a tensor node is conceptually an array figuring in computations, that might or might not have different, distinct or shared, memory array instances in different contexts. A tensor node can be virtual, with no array instances. If it is not virtual, different devices that compute using a tensor node will necessarily store different memory arrays.
  - `Indexing`: a representation and support for indexing into arrays, centered around `projections` from which for loops over arrays can be derived.
  - `Assignments`: the user-facing high-level code representation centered around accumulating assignments.
  - `Low_level`: an intermediate for-loop-based code representation.
- "Backends": the interface and implementations for executing code on different hardware.
  - `Backend_intf`: the user-facing interface.
    - To simplify backend-generic code, the core types are plain records, paremeterized by implementation-dependent components `'buffer_ptr` (memory pointers for arrays), `'dev` (interfacing with devices), `'runner` (interfacing with execution streams), `'event` (synchronizing between execution streams).
    - The final signature `Backend` is split into pieces to avoid signature duplication when combining device-specific implementations with code that can potentially be shared by different backend implementations.
  - `Backend_impl`: the signatures for backend implementations; and components that can be shared across implementations and are not solely used from the `Backends` module.
  - `C_syntax`: the code shared by backends that produce textual C or C-like program representations.
  - Device-specific implementations, currently: `Cc_backend` (any C compiler via text), `Gcc_backend` (GCC compiler via libgccjit), `Cuda_backend` (Nvidia CUDA via the driver API and NVRTC).
  - `Schedulers`: For CPU backends, there are two axes of variation: how to implement single-core computation, and how to parallelize computations across cores. Currently `Schedulers` contains the CPU parallelization implementations -- might be split into more files in the future.
  - `Backends`: collects all user-facing backend implementations -- modules of type `Backend_intf.Backend`. Currently, it covers:
    - Components shared across backends that build on top of device / hardware / external compiler-specific code:
      - The functor `Add_device` combines a single-core CPU implementation with a scheduler, and brings them on par with the device-specific implementations.
      - The functor `Raise_backend` converts any backend implementation relying on the `Low_level` representation (all backends currently), to match the user-facing `Backend_intf.Backend` interface (which relies on the high-level `Assignments` representation).
        - The functor `Add_buffer_retrieval_and_syncing` (used by `Raise_backend`) converts (array pointer) `buffer_ptr`-level copying opeations, to tensor node level, and adds per-tensor-node stream-to-stream synchronization.
    - Putting the above together with the device specific implementations, and exposing the resulting modules to the user via backend names.
      - It also exposes a couple of backend-generic functions:
        - `reinitialize` a backend,
        - `finalize` a context (freeing all of its arrays that don't come from its parent context).

### Shared (relocatable) compilation, batch compilation

Shared (relocatable) compilation, with `~shared:true`, improves compilation efficiency, because code can be compiled once for use on multiple devices (in multiple contexts). It also improves debugging convenience, by generating fewer debugging artifacts. A potential downside is slightly less efficient computations.

Batched compilation has similar benefits, especially in producing fewer debugging artifacts. The compilation might also be slightly more efficient since the compiler needs to be invoked fewer times. While `~shared:true` compiles _one routine for many devices_, batched compilation and linking process _many routines for one device_ at once.

## Tensor nodes, arrays, memory properties

OCANNL classifies tensor nodes according to their memory properties:

 ```ocaml
(** A possible algorithm for deciding sharing within a single device:
    - If a tensor node is read-only for a context, and not otherwise recorded, it is stored as a
      cross-stream sharing candidate.
    - If a cross-stream sharing candidate is read-only for another context, whose parent does not
      have the corresponding array (i.e. it is a different stream), it is recorded as cross-stream
      shared, and the same array is reused.
    - If a tensor node is writable by a context, and it is not cross-stream shared, it is marked as
      non-cross-stream, the array is removed from cross-stream sharing candidates if present. If it
      is cross-stream shared, it is recorded as owned by the corresponding stream. It is an error if
      the node was already owned by a different stream.

    If a tensor node is shared cross-stream, within-device copying is a NOOP as source and
    destination pointers are in that case identical. *)
type sharing =
  | Unset
  | Per_stream  (** The tensor node has separate arrays for each stream. *)
  | Shared_cross_stream  (** The tensor node has a single array per device. *)

type memory_type =
  | Constant  (** The tensor node does not change after initialization. *)
  | Nonconstant  (** One of: [Changed_on_devices], [Volatile]. *)
  | Changed_on_devices of sharing
      (** The tensor node will only change on host via a [to_host] call. *)
  | Volatile
      (** The tensor node will only change on any device via a [from_host] call possibly followed by
          [device_to_device]. *)

type memory_mode =
  | Effectively_constant  (** Either [Hosted Constant], or a subset of [Virtual]. *)
  | Virtual  (** The tensor node's computations are inlined on a per-scalar basis. *)
  | Never_virtual  (** One of: [Local], [On_device], [Hosted]. *)
  | Local
      (** The full tensor node is cached for the duration of a computation but not persisted across
          calls to compiled functions. It is not available for merging across devices. *)
  | Device_only  (** One of: [Local], [On_device]. *)
  | On_device of sharing
      (** The tensor node is stored on the devices that compute with it and persisted across
          function calls. It is available for merging across devices (for devices that support
          merging / P2P), but not (directly) for visualization or storing to disk. *)
  | Materialized  (** One of: [On_device], [Hosted]. *)
  | Hosted of memory_type
      (** The tensor node is stored in a globally addressable memory, in addition to on devices
          where it is computed with (or only on the host and not on the device, for some backends).
          It is available for all operations, and visible to OCaml programs as an {!Ndarray} (the
          optional [array] of {!t}). *)
 ```

 `Tnode.update_memory_mode` verifies consistency of the updates of these modes. Currently, these properties are either set explicitly (directly or indirectly) by the user, or determined by the `Low_level` analysis and optimization process and refined regarding sharing by the context array allocation process (generic across backends). Moreover, the `Tensor` module can influence whether the mode is constant (`Tensor.number`, `Tensor.ndarray`) or non-constant (`Tensor.param`).

A backend can make more refined distinctions, for example a `Local` node in CUDA could optionally be shared across threads of a block.

Contexts track (or store) the on-device arrays corresponding to tensor nodes. Contexts form a hierarchy: linking takes a parent context and outputs a child context. Related contexts that use a tensor node must use the same on-device array for the tensor node. If two unrelated contexts are on the same device, i.e. have a common ancestor, and use the same tensor node that is not part of the most recent common ancestor, the behavior is undefined.

## Typical details of a backend implementation

During the compilation process, the old context cannot be available if the backend supports shared compilation. A backend may for simplicity not suport shared compilation, i.e. ignore `~shared:true` and postpone compilation to the linking phase. Currently, the CUDA backend ignores `~shared:false` and always generates context-and-device-independent kernels, that refer to context (i.e. global) arrays via parameters.

We use keys of the `Low_level.traced_store` containers assuming that they are precisely the tensor nodes used in the compiled code -- and the `Virtual` nodes are the ones optimized-away. The context can contain nodes from the parent context corresponding to tensors only needed by parent or ancestor context's computations. The `get_ident` function (e.g. provided by `C_syntax`) returns a human-readable identifier that's un-ambiguous in the context of the compiled code (shared within `compile_batch`).

Conventionally, the compilation implementation is split into three functions / layers:

- `compile_main` does the bulk of translating a `Low_level.t` into the backend-specific code.
- `compile_proc` populates the parameters in the function header, fills-in the function's initialization section that sets up local arrays, clears these arrays (whether local or global) that need to be zero-initialized or reset to zero, appends the `compile_main` code.
- `compile`, resp. `compile_batch`, compiles the function, resp. functions, into an executable object/file or assembly object/file.
  - On same-machine CPU backends, these functions also dynamically load (if applicable) the code (since there's shared program memory for all cores) -- `compile_batch` includes the same resulting (dyn-loaded) object in the code output for all functions.
  - On GPU-like backends, we cannot load the code at compile time. For example, the CUDA driver API function `cuModuleLoadDataEx` loads the module into _the current context_, which is device-specific, so it must be called from within `link` or `link_batch`.
    - GPU-like backends necessitate distinguishing between `link` and `link_batch`, to prevent the same code from being loaded as multiple modules.

The `C_syntax` functor returns the `compile_proc` function for use by `compile` and `compile_batch` of the backends.

### Conditionally emitting the tracing debugger code

Backends should support logging some of the computations when `Utils.settings.debug_log_from_routines` is set. Obviously, setting this debug information only makes sense for tiny models / computations. For GPU backends, cleanly logging from more than one thread per device would add too much complexity, so we restrict logging to a single thread (`blockIdx = 0, threadIdx = 0`).

We output a log line only for comments and array assignments (corresponding to non-virtual node computations), but we log the computed expression structure: the indices, array values, and inline computation values. For simplicity and conciseness, we don't log the structure of inlined computations. Comments determine the nesting of the `ppx_minidebug` entries: when lowering a `Block_comment`, `Assignments.to_low_level` outputs a `Comment "end"` at the end of a block. Comments are prefixed with `COMMENT:`. For assignments, we also log the debug information stored in the `Set` construct -- it's the computed expression translated into the `%cd` syntax. For processing, the debug information is prefixed by `#` and has endlines replaced by `$`. These structural prefixes / infixes are parsed out by `Utils.log_trace_tree`.

#### Tracing via `stdout`

Since the CUDA backend can only log to the standard output, it passes `let logs_to_stdout = true` to `C_syntax`. This uses `printf`, and prefixes each log line with a kernel run ID. When postprocessing the logs, each run extracts its own log lines. Simultaneous logging from multiple CUDA devices should still be clean -- without interleaving lines -- because the driver is supposed to dump the logs to standard output at device synchronization points.

When using the default stream, CUDA would predictably write to the standard output at context synchronization only. Unfortunately, it does not appear to be the case with asynchronous streams. [Despite the assurance from the documentation, output happens in between CUDA calls...](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output) To remedy this, we implement a `stdout` filtering scheme (function `Utils.capture_stdout_logs`), where all output is captured, tracing lines extracted, and other output printed on the original `stdout`.

## Synchronization and data transfers

OCANNL expects backends to implement FIFO queue scheduling, and an event mechanism for synchronizing between streams (and ideally devices), matching the CUDA specification. On top of events, OCANNL implements per-tensor-node synchronization, using the fields `stream_working_on` of the device record, and `queried_work_for` of the stream record.

```ocaml
...
  stream_working_on : (int * 'event) option Hashtbl.M(Tnode).t;
      (** The stream that most recently has been updating the node, and the associated update
          completion event. An entry for a tensor node is only populated when
          {!field-queried_work_for} is also populated. *)
...
  queried_work_for : 'event option Hashtbl.M(Tnode).t;
      (* The completion event for updating the node via this stream, if any. Only existing entries
         are updated, and an entry is populated when {!work_for} is called for the first time on the
         tensor node. *)
...
  val work_for : context -> Tnode.t -> event option
  (** If the tensor node is in the context, returns the event indicating if currently running or
      scheduled computations modifying that node on the context's stream have completed.

      NOTE: [work_for ctx tn], if work tracking was not yet registered for [tn], will register work
      tracking for [tn] and return the [all_work] event for [ctx]'s stream. *)
```

Besides routines, calling `from_host`, `to_host`, `device_to_device` from a backend puts the corresponding tasks on the device's queue. Both invoking a routine and calling these copying functions will perform the necessary event creations and synchronizations to ensure that when scheduling writing into an array precedes scheduling reading from it, the actual writing also precedes the actual reading.

### Data transfers

OCANNL supports asynchronous data transfers by embedding them in the scheduling mechanism.

OCANNL provides explicit _merge buffers_ for performing those tensor node updates, where different versions of a tensor node from two streams feature in the same computation. The `%cd` syntax for using merge buffers is via the `.merge` pseudo-field. For example, the code for merging gradients might be: `[%cd p.grad =+ p.grad.merge]`. In the current design, there's at most one merge buffer per stream, and the memory is reused for merging different nodes. We keep track of the specific tensor node that was scheduled to occupy this buffer in the stream, and the merge node expected by the linked code, so that we can detect mismatches at scheduling time.

The interface exposes two modes of utilizing merge buffers. The `Streaming` mode relies in some way on the array from the source context. Currently, this simply means using the source array (buffer) pointer, and the CUDA backend falls back to using `~into_merge_buffer:Copy` when the source and destination contexts live on different devices. The `Copy` mode uses physical arrays to back merge buffers. The merge buffer array (one per stream) is resized (grown) if needed to fit a node's array.

Currently, OCANNL does not support merge buffers for `from_host` transfers. But it might in the future. Currently, combining `to_host` and `from_host` is the only way to make different backends cooperate, and that requires `from_host ~into_merge_buffer` to adapt single-backend design patterns.
