# Execution Dependency Tracking (Mirroring Compilation Dependencies)

## Motivation

README.md line 90 and ROADMAP.md line 169 list this as a v1.0 ergonomics item: "Similarly to how contexts track initialization dependencies for compilation, we should also track them for execution."

Today, users must call routines in the correct order themselves. The `Context.run` function (in `arrayjit/lib/context.ml`) already checks that a routine's input nodes are marked as initialized before execution, and marks outputs as initialized afterwards. However, this tracking lives only at the `Context` module level and applies to the simplified API. At the lower `Backends` level (`sync_routine` in `arrayjit/lib/backends.ml`), routines are scheduled onto FIFO queues with event-based synchronization for cross-stream data hazards, but there is no enforcement of routine-level ordering within a single stream.

The gap: if a user schedules routine B (which reads tensor X) before routine A (which writes tensor X) on the same stream, the system silently computes with stale or uninitialized data. The `Context` module catches this for `initialized_nodes` but not for execution-order violations between already-initialized tensors that are being updated.

## Current State

**`Context` module** (`arrayjit/lib/context.ml`, `context.mli`):
- `t` carries an `initialized_nodes : Set.M(Tn).t` tracking which tensor nodes have been initialized (via compilation, `run`, or `init_from_host_deprecated`).
- `run` checks `Set.diff routine.inputs ctx.initialized_nodes` and fails if any inputs are missing. After execution, it unions `routine.outputs` into `initialized_nodes`.
- This is a compile-time / first-run dependency check only. It does not track ongoing read-after-write ordering between routines that update overlapping tensors across multiple calls.

**`Backend_intf.routine`** (`arrayjit/lib/backend_intf.ml` lines 51-61):
- Each routine carries `inputs : Set.M(Tnode).t` (read-only and read-before-write nodes) and `outputs : Set.M(Tnode).t` (written-to nodes).
- These sets are computed at link time from the lowered IR (`Low_level.input_and_output_nodes`).

**`sync_routine`** (`arrayjit/lib/backends.ml` lines 207-235):
- Handles cross-stream synchronization via events: waits on `shared_writer_streams` for input nodes, records `updating_for` events for output nodes.
- Handles `from_host` transfers for hosted inputs when `automatic_host_transfers` is enabled.
- Does NOT enforce any ordering of routines within the same stream beyond FIFO queue order.

**Training loop** (`lib/train.ml`):
- `run_once`, `forward_once`, `update_once` all thread the `Context.t` through `compile` then `run`, which enforces initialization order.
- Multi-step training (init params, then forward, then backprop) works because `Context.t` is threaded linearly. But nothing prevents a user from reusing a stale context or calling routines out of order at the backend level.

## Proposed Change

Extend the execution dependency model so that:

1. **Routine-level write-after-read and read-after-write ordering is tracked.** Each `Context.t` (or equivalent state) should maintain a per-node "last-written-by" stamp (e.g., a monotonic counter or routine identifier). When a routine is scheduled, the system verifies that all its input nodes were last written by a routine that has already been scheduled (or were externally initialized). This catches same-stream ordering bugs.

2. **Independent routines are identifiable.** Two routines are independent if their input/output node sets do not overlap (no WAR, RAW, or WAW hazards). The system should be able to answer "can these two routines run in any order?" -- useful for future parallel scheduling even within a single-device context.

3. **Incorrect ordering produces a clear error.** When a routine is scheduled whose inputs depend on a not-yet-scheduled write, the system should raise an error with the names of the missing dependencies and the routine that was expected to produce them.

### Acceptance Criteria

- Execution dependencies are tracked explicitly alongside compilation dependencies in the context or routine type.
- The system automatically enforces correct execution order (at minimum, a runtime check at scheduling time).
- Incorrect ordering is detected and reported as an error with actionable diagnostics (which nodes are missing, which routine should have been run first).
- Independent routines can be identified programmatically (query or data structure), enabling future parallel execution.

### Edge Cases

- **Routines that both read and write the same node** (e.g., in-place update `x += ...`): the node appears in both `inputs` and `outputs`. The dependency must be on the *previous* write, not the current routine's own write.
- **Re-running the same routine** (e.g., in a training loop): the system must allow a routine to consume its own prior outputs as inputs in the next iteration. The "last-written-by" tracking needs to handle cyclic re-execution gracefully.
- **`init_from_host_deprecated` and `from_host` transfers**: these write to nodes outside the routine mechanism. They must also update the dependency tracking state.
- **Cross-stream execution**: the existing event-based synchronization in `sync_routine` already handles cross-stream data hazards. The new tracking should complement, not replace, that mechanism.

## Scope

**In scope:**
- Extending `Context.t` (or the backend-level state) with execution dependency tracking beyond `initialized_nodes`.
- Adding a scheduling-time check that validates routine ordering.
- Providing a query mechanism to determine routine independence.
- Updating `from_host`, `to_host`, and `device_to_device` paths to participate in dependency tracking.

**Out of scope:**
- Automatic reordering or parallelization of routines (future work building on the dependency graph).
- Changes to the existing cross-stream event synchronization mechanism.
- Static scheduling or program search (that is the v0.9 milestone).

**Dependencies:**
- Relates to `watch-ocannl-README-md-b61f3434` (the related task listed in frontmatter).
- No hard blockers. This is a self-contained ergonomics/safety improvement.
