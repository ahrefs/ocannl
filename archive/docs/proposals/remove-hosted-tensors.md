# Remove Hosted Tensors: Context-Based Tensor Access

## Motivation

The current `Tnode.t` type carries an `array : Nd.t option Lazy.t` field that maintains a host-side copy of tensor data in globally addressable memory. This "hosted tensor" model conflates tensor identity with data residency, making the memory model unnecessarily complex and coupling OCaml-level value access to a specific memory layout. The `Hosted` memory mode adds a large surface of variants (`Unset_hosted`, `Constant`, `Nonconstant`, `Changed_on_devices`, `Volatile`) and a complex state machine of mode transitions.

Removing hosted tensors and requiring context-based access simplifies the architecture, makes device-to-host transfers explicit, and prepares OCANNL for buffer eviction / dynamic memory management in the future. It also produces a cleaner frontend API needed for the ICFP 2026 workshop paper.

GitHub issue: [ahrefs/ocannl#333](https://github.com/ahrefs/ocannl/issues/333)

## Current State

### The `array` field and host-side copies

`Tnode.t` (arrayjit/lib/tnode.ml, line 73-74) stores:
```ocaml
type t = {
  array : Nd.t option Lazy.t;  (* host-side materialized copy *)
  ...
}
```

All direct value access (`get_value`, `set_value`, `get_exn`, `get_values`, `set_values` at lines 507-844) reads from or writes to `Lazy.force t.array`, failing if the array is `None`.

### The `.@` operators

`Operation.At` (tensor/operation.ml, lines 11-31) provides syntactic sugar:
```ocaml
let ( .@{} ) t = Tn.get_value t.Tensor.value
let ( .@{}<- ) t = Tn.set_value t.Tensor.value
```
These are used in ~194 occurrences across 34 files (tests, training examples, library code).

### The `Hosted` memory mode

`memory_mode` (tnode.ml, lines 47-65) includes `Hosted of memory_type`, where `memory_type` has five variants. The `update_memory_mode` function (lines 306-352) contains extensive case analysis involving `Hosted` modes. (Note: `update_memory_sharing` was removed in the streams cleanup.)

Helper predicates: `is_hosted_force`, `is_in_context_force`, `known_volatile` all pattern-match on `Hosted` variants. (Note: `known_shared_cross_streams` was removed in the streams cleanup.)

### Train.ml hosted helpers

`train.ml` exposes:
- `set_on_host` (line 63): marks a tnode as `Hosted Changed_on_devices` or `Hosted Volatile` (note: `Changed_on_devices` is no longer parameterized after streams cleanup)
- `set_hosted` (line 69): marks as `Hosted Constant` or `Hosted Changed_on_devices`
- `every_non_literal_on_host` (line 180): sets all embedded, unspecified nodes as hosted
- `forward` (line 75), `grad_update` (line 84): call `set_hosted` on root values
- `to_routine` (line 186), `init_params` (line 208), `run_once` (line 251), `forward_once` (line 278): all accept `?(hosted = true)` parameter and call `set_hosted` on collected nodes

### Backend interface

`backend_intf.ml` defines `from_host` (copies host array to device) and `to_host` (copies device to host array). `Context` wraps these; `init_from_host_deprecated` (context.ml, line 156) is already marked as deprecated.

## Proposed Change

Remove the `Hosted` memory mode and the `array` field from `Tnode.t`. All tensor value access goes through a context, with on-demand device-to-host transfers.

### Core type changes

1. **Remove `array` from `Tnode.t`**: The `array : Nd.t option Lazy.t` field is deleted. No host-side copy is stored in the tensor node.

2. **Remove `memory_type` and simplify `memory_mode`**: Delete the `Hosted of memory_type` variant and the entire `memory_type` type. `On_device` becomes the primary materialized mode. The `Materialized` umbrella mode collapses to just `On_device`.

3. **Remove or repurpose `Ndarray` usage**: `Ndarray` stays as a utility for creating data buffers passed to `from_host`, but is no longer stored inside `Tnode.t`.

### New context-based access API

Replace `Tn.get_value` / `Tn.set_value` (which read from the `array` field) with context-aware operations. Two design options exist per the issue discussion:

- **Option A (no caching)**: `Context.get_value ctx tnode indices` performs a device-to-host copy into a temporary buffer, reads the value, discards the buffer. This is the simplest design and matches the author's final comment: "we will make nothing hosted."

- **Option B (mutable cache)**: Replace `array` with `mutable array_cache : Nd.t option`. On first context-based read, populate the cache; allow eviction. This is more practical for interactive use but adds complexity.

The author's comments suggest Option A as the target, with the printing trick (below) handling the most common use case.

### Printing via on-demand assignment

From the issue: tensor printing uses `[%cd "for_print" =: t_to_print]` to create a temporary device-to-host copy when the tensor is not already materialized on the host. A cache of "for-print" tensor nodes avoids re-creating assignments on every print call. `Tensor.print` (off by default) and `Train.printf` (on by default) control this behavior.

### Train.ml cleanup

- Delete `set_on_host`, `set_hosted`, `every_non_literal_on_host`
- Remove `?(hosted = true)` parameter from `to_routine`, `init_params`, `run_once`, `forward_once`
- The `forward` and `grad_update` functions stop calling `set_hosted`
- `init_params` (line 225-228) no longer needs the `init_from_host_deprecated` fallback

### `.@` operator migration

The `.@` operators cannot work without a context. Options:
- Remove them entirely, replace all 194 call sites with `Context.get_value ctx ...`
- Redefine them to require a context parameter (less ergonomic as OCaml custom operators have fixed arity)
- Provide a `with_context` scope that implicitly threads a context for `.@` access (e.g., via a thread-local or module-level mutable)

The choice affects all 34 files and the workshop paper examples.

## Scope

### In scope

- Removing `array` field from `Tnode.t` and `Hosted` from `memory_mode`
- Removing `memory_type` type entirely
- Simplifying `update_memory_mode` case analysis (note: `update_memory_sharing` was already removed in the streams cleanup)
- Removing `set_on_host`, `set_hosted`, `every_non_literal_on_host` from `train.ml`
- Removing `?(hosted = ...)` parameters from `to_routine`, `init_params`, `run_once`, `forward_once`
- Implementing context-based value access (the new `get_value`/`set_value`)
- Implementing on-demand printing via the `[%cd "for_print" =: t]` trick
- Updating all test files (34 files, ~194 occurrences)
- Removing or updating `get_exn`, `has`, `get_values`, `set_values` in `tnode.ml`
- Updating `devices_not_lagging_host` tracking logic

### Out of scope

- Full removal of the `Ndarray` module (it is still needed for data loading buffers and backend code generation with its 12 precision variants)
- Buffer eviction / dynamic lifetime management (future work enabled by this change)
- Changes to `from_host` / `to_host` backend interface signatures (they continue to work, but `to_host` becomes the primary way to retrieve values)
- Unified memory optimization for Apple Silicon (orthogonal; `use_host_memory` flag stays)

### Dependencies

- **watch-ocannl-README-md-b61f3434** (deprecated streams cleanup): completed; the `sharing` type and `On_device of sharing` parameterization have been removed
- **gh-ocannl-373** (tensor saving/loading): depends on this task's design for on-demand retrieval
