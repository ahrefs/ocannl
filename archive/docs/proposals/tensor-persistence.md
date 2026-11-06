# Tensor Saving, Loading and Restoring

## Motivation

OCANNL has no way to persist tensor data to disk and load it back. This blocks checkpointing during training, transfer learning, and model distribution. The commented-out `save_params`/`restore_params` in `lib/train.ml` (lines 44-62) used Npy.Npz but was never completed and has been disabled through API evolution.

GitHub issue: https://github.com/ahrefs/ocannl/issues/373

## Current State

**No persistence API exists.** The only remnant is commented-out code in `lib/train.ml` that serialized tensors by label name into `.npz` format.

**Relevant infrastructure that does exist:**

- `Ndarray.retrieve_flat_values ?padding arr` (`arrayjit/lib/ndarray.ml:401`) -- reads an ndarray as a `float array`, respecting padding boundaries
- `Ndarray.set_flat_values ?padding arr values` (`arrayjit/lib/ndarray.ml:422`) -- writes an ndarray from a `float array`
- `Ndarray.create_array ~debug prec ~dims ~padding` -- allocates a new ndarray
- `Ndarray.size_in_bytes` (`arrayjit/lib/ndarray.ml:324`)
- `Ops.prec` has `sexp_of_prec` / `prec_of_sexp` (`arrayjit/lib/ops.ml:75-105`)
- `Ops.axis_padding` has `[@@deriving sexp]` (`arrayjit/lib/ops.ml:877`)
- `Tnode.t` has fields: `id : int`, `label : string list`, `prec`, `dims`, `padding`, `array`, `memory_mode`, `prepare_read`, `prepare_write`
- `Tnode.registry` -- weak hash table for all live tnodes (`arrayjit/lib/tnode.ml:604`)
- `Tnode.find ~id` -- looks up a tnode in the registry by integer id (`arrayjit/lib/tnode.ml:761-780`)
- `Tensor.tn_set = Set.M(Tn).t` (`tensor/tensor.ml:10`)

**Namespace status (#372):** The GitHub issue is closed, but namespaces are NOT yet in the codebase. `Tnode.t.id` is still `int`, not `{ namespace : string; s_id : int }`. The persistence implementation must either:
1. Land after namespaces are actually merged, or
2. Use integer IDs now, with a migration path to namespaced IDs later.

Option 2 is pragmatic -- tensors can be identified by `(label, id)` pairs in checkpoint files, and the `load` function's `?prefix_namespace` parameter can be deferred until namespaces exist.

**Device transfer:** `prepare_read` triggers device-to-host sync; `prepare_write` triggers host-to-device sync. The `set_on_host` helper in `lib/train.ml:63` sets the memory mode appropriately.

## Proposed Change

Add three functions -- `save`, `load`, `restore` -- as described in the issue. These should live in a new `persistence.ml` module (or in `train.ml` alongside the existing commented-out code).

### `save ~appending t_set path`

Writes tensor data to a checkpoint file. When `~appending:true` and the file exists, replaces overlapping tensors but keeps others already in the file. When `~appending:false`, overwrites the file entirely.

### `load path` (initially without `?prefix_namespace`)

Reads tensors from a checkpoint file, creates new tnodes, registers them, and returns the resulting `tn_set`. Raises if any loaded tensor clashes with an existing tnode in the registry. The `?prefix_namespace` parameter should be added once #372 namespaces are merged.

### `restore t_set path`

Takes a set of existing tnodes, finds their data in the checkpoint file by ID, and updates their hosted buffers. Raises if a tensor from `t_set` is missing in the file, or if dimensions/precision don't match.

### File format

S-expression header + contiguous binary data:

```
[4 bytes: header length as uint32]
[header bytes: S-expression with version, tensor metadata (id, label, prec, dims, padding, offset, size_bytes)]
[tensor binary data concatenated]
```

This leverages the existing `sexplib` dependency. Tensor data is stored in native precision format (no conversion to float64), preserving precision for Fp8/Bfloat16/Half types and reducing file size.

### Acceptance criteria (from issue)

- `save` with both append and overwrite semantics
- `load` with registry clash detection (namespace prefixing deferred to post-#372)
- `restore` with dimension/precision verification
- Round-trip correctness across all 12 precision types
- File format includes version number for forward compatibility
- No regression in existing tests

### Edge cases

- Empty tensor sets: `save` creates a valid empty checkpoint; `restore` is a no-op; `load` returns an empty set
- Dimension or precision mismatch on `restore`: raise, don't silently truncate
- Virtual/Local tnodes in `t_set`: filter or raise (they have no hosted data)
- Appending to a large file requires reading+rewriting non-overlapping tensors (correct but expensive)
- Padding: persist only logical (unpadded) values via the `?padding` parameter of `retrieve_flat_values`

## Scope

**In scope:**
- The three core functions (`save`, `load`, `restore`)
- Checkpoint file format (S-expression header + binary)
- Registry integration for clash detection
- Device-to-host / host-to-device transfers around save/load/restore
- Basic round-trip tests

**Out of scope:**
- `?prefix_namespace` on `load` (blocked on #372 namespaces actually landing)
- Adapting to removal of hosted memory mode (#333 -- still open)
- Cross-framework interop (numpy, safetensors, etc.)
- File locking or concurrent access
- Precision conversion on load (e.g., loading float32 into float16 model)
- Streaming/lazy loading for very large models

**Dependencies:**
- Soft dependency on #372 (namespaces): the `?prefix_namespace` parameter is deferred, but the file format should reserve space for namespace metadata so migration is smooth
- Future impact from #333 (remove hosted memory): will require changing data access from `tn.array` to context-mediated transfers
