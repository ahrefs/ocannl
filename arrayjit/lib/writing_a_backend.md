# The Anatomy of an OCANNL Backend

## Design around compiling and running code

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

