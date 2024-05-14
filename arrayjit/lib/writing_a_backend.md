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

