# Proposal: Introduce Training Checkpointing

Tracked by: https://github.com/ahrefs/ocannl/issues/96

## Status update (2026-06-12)

- Issue #96 is OPEN, milestone **v1.0** (due 2026-06-30). Not started: there is still no
  `Train.checkpoint*` API, no `lib/checkpoint.ml`, and the old npz-based `save_params` /
  `restore_params` code remains commented out at `lib/train.ml:44-62`.
- The dependency story still holds exactly as written: `lib/persistence.{ml,mli}` (#373) is
  unchanged — `save : appending:bool -> Tensor.tn_set -> string -> unit`,
  `load : ?prefix_namespace:string -> string -> Tensor.tn_set` (non-empty prefixes still
  rejected, reserved for #372), `restore : Tensor.tn_set -> string -> unit`.
- Optimizer state is still anonymous: `Train.sgd_one` (`lib/train.ml:99-108`) still introduces
  `sgd_momentum` via an inline `%cd` declaration with no first-class handle, and
  `example_train_result` is still at `lib/train.ml:183`. Open question 4 (explicit
  `optimizer_state : tn_set`) remains the main design decision.
- Minor line drift in `tensor/tensor.ml`: `random_seed` ref is now at line 120,
  `set_random_seed` at 747, and the `Tn.update_prec res.value Ir.Ops.uint4x32` call at
  line 752 (was 744). No semantic change.
- Repo-wide changes since April (multi-stream backend-layer cleanup, `device_to_device`
  returning a transfer routine, label→basis rename) do not touch this proposal's seams;
  the checkpointing design works at the host-buffer level via `Persistence`.
- Remains to do: everything in the Approach section (optimizer-state refactor, checkpoint
  save/load API, RNG handling, `docs/persistence_format.md`, `test/training/` integration
  test). New `test/training/` examples (bigram, transformer_names, mnist_conv, ...) provide
  ready-made small training loops for the round-trip test.

## Goal

Provide a `Train.checkpoint` API that saves and restores all global training
state -- model parameters, optimizer state, RNG state, and training-loop
metadata -- so that a training run can be interrupted and resumed at a later
time on the same machine and produce results indistinguishable from an
uninterrupted run. Build on top of the tensor persistence primitives
(`lib/persistence.ml`) delivered by #373; keep the on-disk layout small,
self-describing, and documented.

## Acceptance Criteria

Falsifier-style; each criterion describes a concrete way the AC fails.

- [ ] **Round-trip determinism (params + optimizer + RNG together).** A test in
  `test/training/` runs SGD-with-momentum on a small fixed problem for
  `N + M` steps in process A. In process B, it runs `N` steps, calls
  `Train.checkpoint_save path`, exits, starts a fresh process C, calls
  `Train.checkpoint_load path`, and runs `M` more steps. Final parameter
  buffers from A and C must agree bit-for-bit (same backend, same precision)
  -- or, if a tight float tolerance is unavoidable for backend-determinism
  reasons, within `1e-6` relative error per element. *Falsifier: any drift
  larger than the stated tolerance fails.*

- [ ] **Optimizer-state restore (momentum-direction probe).** A separate test
  configures `Train.sgd_update ~momentum:0.9` and inspects the
  `sgd_momentum` tensor associated with one parameter. The `(N+1)`-th update
  vector is recorded across the checkpoint boundary; the value computed in
  the restart-from-checkpoint run must equal the value computed in the
  no-restart run. *Falsifier: if checkpointing only saves params (a known
  regression mode), the `(N+1)`-th update vector differs because momentum is
  zero-initialized after restart.*

- [ ] **RNG round-trip.** After checkpoint save and load, the next tensor
  initialized via a randomized op (or the next sample drawn from
  `Tensor.get_random_seed ()`) must equal the value that the uninterrupted
  run would have produced at the same step. The test asserts this directly
  by drawing one batch of random numbers post-restore and comparing to a
  reference. *Falsifier: dropping `random_seed` from the saved set, or
  saving it as `single` precision instead of `Uint4x32`, fails the
  comparison.*

- [ ] **Loop-metadata round-trip.** The checkpoint records and restores the
  current epoch index, batch index within the epoch, learning-rate schedule
  position, and the running loss histories (`rev_batch_losses`,
  `rev_epoch_losses`). After restore, a single call into the
  `example_train_result` builder produces matching `learning_rates` /
  `rev_*_losses` lists. *Falsifier: missing any of these makes the lists
  diverge after one more step.*

- [ ] **Format documented in-tree.** A new section
  `docs/persistence_format.md` (or an appended section to
  `docs/proposals/tensor-persistence.md`) describes the on-disk layout of a
  training checkpoint: which entries are tensor payloads (handed off to
  `Persistence`), and which are scalar/list metadata stored in the
  S-expression sidecar. The section is referenced from `lib/train.mli`
  doc-comment of the public checkpoint API. *Falsifier: deleting the file
  or leaving it empty makes the doc test (a small `grep` regression check
  in CI, or simply a presence check inside the integration test) fail.*

- [ ] **Integration test added under `test/training/`** with a
  `.expected` file so that CI regression-tests the round-trip determinism
  AC in headless mode, on the default backend (CC). *Falsifier: any
  divergence in the recorded loss trajectory fails the test.*

## Context

### Module-name modernization

The original 2024 GitHub issue references obsolete modules `Node`, `NodeUI`,
`Formula`, `Session`, and stdlib `Random`. The current OCANNL codebase has
moved to:

| Issue (2024) | Current (2026) |
|---|---|
| `Node` | `Ir.Tnode` |
| `NodeUI` | (removed; tensor labels carry the role) |
| `Formula` | `Ocannl_tensor.Tensor`, `Ocannl_tensor.Operation` |
| `Session` | `Train`, `Context` (`arrayjit/lib/context.ml`) |
| stdlib `Random` | `Ocannl_tensor.Tensor.random_seed` (a `Uint4x32` tnode) |

### Dependencies (verified 2026-04-30)

- **#373 (tensor save/load/restore)** -- DONE. `lib/persistence.{ml,mli}`
  exposes:
  - `save : appending:bool -> Tensor.tn_set -> string -> unit`
  - `load : ?prefix_namespace:string -> string -> Tensor.tn_set`
  - `restore : Tensor.tn_set -> string -> unit`

  Format is a length-prefixed S-expression header (`version`, list of
  `tensor_meta`) followed by contiguous binary payloads in native precision.
  `save` calls `Tn.do_read` per tnode to sync device-to-host;
  `restore` clears `prepare_read` and calls `Tn.do_write` to mark devices
  stale so the next access re-uploads from host. `appending:true` merges
  with an existing file, replacing entries by ID.

- **#372 (tensor-ID namespaces)** -- DONE. The `?prefix_namespace` parameter
  is reserved in the `load` signature but currently rejects non-empty
  prefixes; this is fine for v1.0 single-process checkpointing and only
  matters when we eventually want to load multiple checkpoints into the
  same process.

### State that needs to be checkpointed

Verified against `/Users/lukstafi/ocannl-staging`:

1. **Model parameters.** Each `Tensor.t` carries a `params : tn_set`
   (`tensor/tensor.mli:22`). The trainable root tensor (typically the loss)
   transitively aggregates them, so `loss.params` is the canonical
   collection seam. Each element is a `Tensor.t` whose `.value` is the
   parameter tnode -- those are the things `Persistence.save` needs.

2. **Optimizer state -- SGD momentum.** `lib/train.ml:99-108` shows the
   current SGD-with-momentum implementation. The `%cd` block
   `{ sgd_momentum } =: ...` constructs a per-parameter tnode labelled
   `"sgd_momentum"` which is automatically embedded in the update
   computation. It has no first-class handle; checkpointing has to either
   (a) walk `loss`'s embedded tnodes after the update has been built, or
   (b) introduce an explicit `optimizer_state : tn_set` accumulator the
   optimizer functions populate. Option (b) is the cleaner long-term
   choice.

3. **RNG state.** `tensor/tensor.ml:120-160` keeps `random_seed : t option
   ref`, lazily initialized by `Tensor.set_random_seed ?seed ()`. The
   underlying tnode is `Uint4x32`-precision (forced via
   `Tn.update_prec res.value Ir.Ops.uint4x32` at line 752 *(Update
   2026-06-12: was 744)*). The seed tnode
   is an ordinary tnode and `Persistence` already handles arbitrary
   precisions.

4. **Loop metadata.** Currently lives in user-defined training loops as
   `int ref`s (`step_ref`, `batch_ref`, `epoch_ref`) and `float list`
   accumulators. The `example_train_result` record
   (`lib/train.ml:183-194`) is the visible contract. There is no existing
   abstraction; the proposal needs to introduce one or document what the
   user is responsible for round-tripping.

### Why the original Tentative Design's pointers needed updating

The Feb-2026 task elaboration cited:

- `~/ocannl/lib/train.ml:44-62` for the commented-out save/restore -- still
  there in `lib/train.ml:44-62`, but no longer the only relevant code: the
  new `lib/persistence.{ml,mli}` (added 2026-03-26 in commits `d67cbeda` /
  `05fbf6f0` / `757c5875`) is what we now build on.
- `~/ocannl/tensor/tensor.ml:118` for `random_seed` -- it is now at
  `tensor/tensor.ml:120` (a few lines drifted); precision is still
  `Uint4x32` (line 752 as of 2026-06-12). No semantic change.
- `Ndarray.t` -- still at `arrayjit/lib/ndarray.ml`, with
  `read_payload_from_channel` / `write_payload_to_channel` already used
  by `Persistence`.

The dependency story holds: #373 is wired up exactly the way the design
assumed, and `Persistence.save`/`restore` are precisely the seam this
proposal sits on.

## Approach

(Sketch only -- the agent doing the implementation has latitude on the
specifics, especially for the questions called out as ambiguities below.)

### 1. Make optimizer state explicit

Refactor `Train.sgd_one` / `Train.sgd_update` so that the per-parameter
`sgd_momentum` tnodes are exposed as a `tn_set` accumulated alongside
`loss.params`. The cleanest shape is probably an extra return value
(`update_code, optimizer_state`) from the optimizer-update builders, or a
mutable `optimizer_state : tn_set` field stored on a new `Train.optimizer`
record. This unblocks future Adam / AdamW / Lion implementations cleanly:
each writes its slot tensors into the same accumulator.

### 2. Define a `Train.checkpoint` module (or sub-namespace of `Train`)

Two operations:

```ocaml
val save_checkpoint :
  path:string ->
  params:Tensor.tn_set ->
  optimizer_state:Tensor.tn_set ->
  loop_state:loop_state ->
  unit -> unit

val load_checkpoint :
  path:string ->
  params:Tensor.tn_set ->
  optimizer_state:Tensor.tn_set ->
  unit -> loop_state
```

where `loop_state` is a small record (`epoch`, `batch_in_epoch`,
`global_step`, `rev_batch_losses`, `rev_epoch_losses`, `learning_rates`,
optionally a user-extensible `extra : Sexp.t`).

`save_checkpoint` writes to two artifacts side-by-side:

- `<path>.tensors` -- tensor payload, written via `Persistence.save
  ~appending:false` over the union of `params`,
  `optimizer_state`, and `[Tensor.(get_random_seed ()).value]`.
- `<path>.meta` -- S-expression file with `loop_state` plus a tiny
  manifest (`tensor_path`, `version`, `created_at`, set of expected tnode
  IDs).

`load_checkpoint` parses the meta sidecar, then calls
`Persistence.restore` over the same set of tnodes (caller must have
constructed the model graph so the parameter tnode IDs line up; #372 namespacing
makes this robust against ID-collision surprises later). Returns the
`loop_state` for the user to seed their training loop with.

### 3. Random-state handling

`Tensor.get_random_seed ()` lazily allocates the seed tnode on first call.
The save path forces the allocation (via `get_random_seed`) so the seed is
guaranteed to exist in the checkpoint. The restore path treats the seed
exactly like any other parameter -- `Persistence.restore` overwrites its
hosted buffer, and `do_write` ensures the device picks up the new value
on next access.

### 4. Documentation

Add `docs/persistence_format.md` (or extend
`docs/proposals/tensor-persistence.md` with a "Training checkpoint
format" section) describing both the `.tensors` payload (delegated to the
existing #373 format) and the `.meta` sidecar shape. Cross-reference from
`lib/train.mli`.

### 5. Integration test

`test/training/test_checkpoint_resume.ml` exercises the round-trip
deterministically on a small linear-regression-with-momentum problem on
the default CC backend. The test forks two child processes (or runs two
sequential `Stdlib.Sys.command` calls into the same `dune exec` binary
with different argv) so that the "fresh process" property is real and not
faked by reusing in-memory state.

## Ambiguities / Open Questions

These are surfaced for elaboration with the user before implementation.

1. **File-format choice.** Three candidates:
   - **Status quo**: reuse `Persistence`'s sexp-header + binary payload
     for tensors, plus a sidecar sexp file for metadata. Zero new
     dependencies; opaque to other tools.
   - **NumPy `.npz`**: requires bringing back an `Npy`-style dependency;
     immediately readable from Python; helpful for transfer-learning and
     debugging. The original commented-out code went this way.
   - **CBOR / MessagePack**: language-neutral, single-file, but adds a
     dependency.
   Recommended default is option 1 (already paid the dependency cost via
   #373), but a Python-readable export tool may be wanted later for the
   GPT-2 inference workflow (#377 already converts in the opposite
   direction).

2. **Cross-backend compatibility.** Should a checkpoint saved on the CUDA
   backend be loadable on the CPU/CC backend (and vice versa) for v1.0?
   `Persistence.restore` works at the host-buffer level, so this should
   be free in principle, but it has to be tested. Promote to AC, or
   defer with an explicit "single-backend only" caveat?

3. **Partial restore for transfer learning.** Out-of-scope for v1.0?
   Issue #96 doesn't require it, and `Persistence.restore` already takes
   a `tn_set` argument so partial restore is mechanically possible --
   but a user-friendly story (e.g. "restore everything except the
   classifier head") wants a per-label filter we don't yet have.

4. **Optimizer-state abstraction shape.** Right now `sgd_momentum` is an
   anonymous `%cd`-introduced tnode. Refactoring optimizers to expose a
   first-class `optimizer_state : tn_set` is the right move, but it
   touches the public `Train` API. Acceptable scope creep for this
   issue, or split into a precursor issue?

5. **Where should `Train.checkpoint` live?** Inside `lib/train.ml`
   alongside `init_params` / `run_once`, or in its own module
   `lib/checkpoint.ml` so that `Train` does not grow further? The
   `Persistence` module already lives at the top level; mirroring that
   pattern argues for `lib/checkpoint.ml`.

6. **`Train.unsafe_reinitialize` interaction.** `Tensor.unsafe_reinitialize`
   resets the registry and the random-seed singleton. Should
   `load_checkpoint` call it implicitly to guarantee a clean slate, or
   require the user to call it before constructing the model graph?
   Implicit is friendlier; explicit is safer.

7. **OCANNL audit pause.** Per harness memory, OCANNL autonomous work is
   paused for the user's hands-on quality audit. This proposal should
   defer to the user; do not auto-launch.

## Out of Scope

- Multi-host / sharded checkpointing.
- Streaming / partial checkpointing (e.g. weight-only at every step,
  optimizer state every N steps).
- Compression of checkpoint files.
- Format migration tooling (versioned upgrades) -- the `version` field is
  reserved in the header, but no migration path is shipped.
- Integration into a CLI / `bin/` entry point. The API is library-only
  for v1.0.
