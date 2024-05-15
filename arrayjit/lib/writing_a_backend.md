# The Anatomy of an OCANNL Backend

## Design around compiling and running code, backend interfaces

Currently, OCANNL integrates new backends via code in [backends.ml](backends.ml). `Backends` introduces the context-specific `routine` type, and has a helper `lower_assignments` that wraps `Assignments.lower` and `Low_level.optimize_proc`. The interface to a `Backend` has `compile` functions, to allow some backends handle assignments directly, instead of using the optimized C-like representation `Low_level.t`.

```ocaml
type lowered_bindings = (static_symbol, int ref) List.Assoc.t  (* in indexing.ml *)

type work = Work of ((module Debug_runtime) -> unit -> unit)  (* in tnode.ml *)

type 'context routine = {
  context : 'context;
  schedule : unit -> work;
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

Backends need to provide the `code` (for compilation result) and `context` types. `code` is some intermediate state between assignments and `context routine`. A backend may postpone doing anything specific until linking, e.g. `code = Low_level.optimized`, or linking may be a no-op, effectively `code = routine`, usually it will fall somewhere in between and depend on whether `~shared:true` is passed. For simple situations, `Backends` has a helper functor:

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

`No_device_backend`s do not themselves deal with the device abstraction. There's the functor `Multicore_backend (Backend : No_device_backend) : Backend` that assigns a device to a domain, and manages the given `No_device_backend` on the domain-based devices. Scheduling a work captures the state of the bindings at that point. Scheduling should never ever block. Running a work on a `No_device_backend` _should block_ (till execution finishes), but it _should not block_ for a proper `Backend` -- it should just launch the work.

```ocaml
module type Backend = sig
  include No_device_backend

  type device

  val init : device -> context
  val await : device -> unit
  val get_device : ordinal:int -> device
  ...
end
```

`Backend.await` synchronizes the device -- waits for all work on the device to finish.

### Shared (relocatable) compilation, batch compilation

Shared (relocatable) compilation, with `~shared:true`, improves compilation efficiency, because code can be compiled once for use on multiple devices (in multiple contexts). It also improves debugging convenience, by generating fewer debugging artifacts. A potential downside is slightly less efficient computations.

Batched compilation has similar benefits, especially in producing fewer debugging artifacts. The compilation might also be slightly more efficient since the compiler needs to be invoked fewer times.

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
    context -> procedure -> context * Indexing.lowered_bindings * (unit -> Tnode.work) * string

  ...
end
```

Contexts track (or store) the on-device arrays corresponding to tensor nodes. Context form a hierarchy: linking takes a parent context and outputs a child context. Related contexts that use a tensor node must use the same on-device array for the tensor node. If two unrelated contexts are on the same device, i.e. have a common ancestor, and use the same tensor node, the behavior is undefined. For CPU backends, the arrays might be stored as:

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
  prec : Ops.prec;
  zero_initialized : bool;
  ...
}
```

The `tn_info`s will be stored inside a compilation state datatype typically called `info` or `info_arrays`. During the compilation process, the new context is not available, and even the old context cannot be available if the backend supports shared compilation. A backend may for simplicity not suport shared compilation, i.e. ignore `~shared:true` and postpone compilation to the linking phase. Currently, the CUDA backend does the opposite, it ignores `~shared:false` and always generates relocatable kernels. This does not require any extra compilation flag, because the kernels refer to context (i.e. global) arrays via parameters. We face two cases:

- Non-trivial `~shared:true`: `tn_info`s are by necessity generated from scratch (either during compilation inside a `get_array`, or at once via `prepare_arrays` as in the `gccjit` backend). If this is the only mode the backend supports, they don't need to be stored.
- Non-trivial `~shared:false`: `tn_info`s must be propagated via the context, because to benefit from not sharing, `tn_info` must include context-specific information (typically a memory pointer to the on-device array).

The `gccjit` backend needs to `prepare_arrays` upfront, because it needs to know the list of parameters of the compiled function before it starts the compilation. This forces the `gccjit` backend to postpone creating the array pointers (via lazy initializers), not because of the context array pointers, but because of the local array pointers (on the function stack) which require knowing the function.

Conventionally, the compilation implementation is split into three functions / layers:

- `compile_main` does the bulk of translating a `Low_level.t` into the backend-specific code.
- `compile_proc` populates the parameters in the function header, fills-in the function's initialization section that sets up local arrays, clears these arrays (whether local or global) that need to be zero-initialized or reset to zero, appends the `compile_main` code.
- `compile`, resp. `compile_batch`, compiles the function, resp. functions, into an executable object/file or assembly object/file.
  - On same-machine CPU backends, these functions also dynamically load (if applicable) the code (since there's shared program memory for all cores) -- `compile_batch` includes the same resulting (dyn-loaded) object in the code output for all functions.
  - On GPU-like backends, we cannot load the code at compile time. For example, the CUDA driver API function `cuModuleLoadDataEx` loads the module into _the current context_, which is device-specific, so it must be called from within `link`.
    - Within `link`, the `cuda` backend first calls `Cudajit.ctx_set_current` (if needed) and only then `Cudajit.module_load_data_ex`.

### Conditionally emitting the tracing debugger code

Backends should support logging some of the computations when `Utils.settings.debug_log_from_routines` is set. Obviously, setting this debug information only makes sense for tiny models / computations, otherwise the log files will explode. For GPU backends, cleanly logging from more than one thread per device would add too much complexity, so we restrict logging to `blockIdx = 0, threadIdx = 0`.

We output a log line only for comments and array assignments (corresponding to non-virtual node computations), but we log the computed expression structure: the indices, array values, and inline computation values. For simplicity and conciseness, we don't log the structure of inlined computations. Comments determine the nesting of the `ppx_minidebug` entries: when lowering a `Block_comment`, `Assignments.to_low_level` outputs a `Comment "end"` at the end of a block. Comments are prefixed with `COMMENT:`. For assignments, we also log the debug information stored in the `Set` construct -- it's the computed expression translated into the `%cd` syntax. For processing, the debug information is prefixed by `#` and has endlines replaced by `$`. These structural prefixes / infixes are parsed out by `Utils.log_trace_tree`.

Since the CUDA backend needs to capture the standard output, it additionally prefixes each log line with a kernel run ID. When postprocessing the logs, each run extracts its own log lines. Simultaneous logging from multiple CUDA devices should still be clean -- without interleaving lines -- because the driver is supposed to dump the logs to standard output at device synchronization points.
