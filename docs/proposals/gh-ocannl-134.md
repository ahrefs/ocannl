# Allow Multiple Virtual Tensors to Share the Same For Loop

## Goal

Enable multiple virtual (inlined) tensors that are computed within the same for loop to all be virtualized, rather than forcing them to materialize. Currently, when two tensors share a loop iterator symbol, both are marked `is_complex = true` in `visit_llc`, which prevents their virtualization. This limits optimization opportunities in cases like element-wise operations where several intermediate tensors are computed in the same loop.

Tracked by: https://github.com/ahrefs/ocannl/issues/134

## Acceptance Criteria

- Multiple virtual tensors within a single for loop are each inlined correctly at their use sites.
- `reverse_node_map` is changed from mapping each symbol to a single `Tnode.t` to mapping each symbol to a set of tnodes, so that shared symbols do not force `is_complex = true`.
- `virtual_llc` handles for loops that contribute computations for multiple virtual tensors: when the loop iterator maps to several tnodes, each virtual tnode's computation is extracted and stored via `check_and_store_virtual`.
- `cleanup_virtual_llc` removes for loops where all contributing tensors have been virtualized, and preserves loops that still have non-virtual residual operations.
- `inline_computation` correctly handles inlining a computation that was extracted from a multi-tensor for loop body (the inlined fragment should contain only the operations for the target tnode, not sibling operations).
- Existing virtual tensor tests continue to pass.
- No performance regression for programs that do not exercise the multi-virtual-tensor-per-loop pattern.

## Context

### The problem

In `visit_llc` (line ~310-321 of `arrayjit/lib/low_level.ml`), a `track_symbol` helper maps each loop iterator symbol to the tnode it belongs to via `reverse_node_map : (Symbol.t, Tnode.t) Hashtbl.t`. When a second tnode uses the same symbol, both are marked `is_complex <- true`. The comment on line 312 explicitly notes: "See TODO(#134): this prevents multiple virtual arrays from sharing for loops."

The `is_complex` flag blocks virtualization at line 428: `virtualize_settings.inline_simple_computations && (not traced.is_complex)` -- a complex tensor with too many accesses becomes `Never_virtual`.

### Current pipeline

```
visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc -> eliminate_common_subexpressions
```

1. **`visit_llc`** (line 256): traces tensor accesses, builds `reverse_node_map` (symbol -> single tnode), marks `is_complex`.
2. **`virtual_llc`** (line 775): walks the LLC tree. For `For_loop` nodes whose index maps to a tnode in `reverse_node_map`, it calls `check_and_store_virtual` to record the entire loop as a computation for that single tnode. Currently only one tnode per loop is handled.
3. **`check_and_store_virtual`** (line 457): validates that a computation block is suitable for inlining (no escaping variables, consistent indices, has setters), then stores it in `computations_table`.
4. **`inline_computation`** (line 623): at each `Get` site for a virtual tnode, substitutes the stored computation inline, filtering to only the operations on the target tnode (via `Tn.equal` checks on `Set`/`Zero_out`/`Get`).
5. **`cleanup_virtual_llc`** (line 860): removes operations on now-virtual tnodes from the original locations -- for loops whose iterator maps to a virtual tnode are entirely removed.

### Key insight

`inline_computation` already filters the stored computation body to extract only operations for the target tnode (lines 702-712: `Set`/`Zero_out`/`Get` with `Tn.equal tn traced.tn`). This means even if the full loop body contains operations for multiple tensors, inlining will correctly select only the relevant subset. The main changes needed are in the bookkeeping layer, not the inlining logic itself.

### Approach sketch

1. **Change `reverse_node_map`** from `(Symbol.t, Tnode.t) Hashtbl.t` to `(Symbol.t, Tnode.t list) Hashtbl.t` (or a set). Remove the `is_complex <- true` marking when multiple tnodes share a symbol.

2. **Update `virtual_llc`**: when a for loop's index maps to multiple tnodes, call `check_and_store_virtual` for each virtual tnode in the list, passing the same loop body. Each tnode gets the full loop body stored as its computation.

3. **Update `cleanup_virtual_llc`**: when a for loop's index maps to multiple tnodes, only remove the loop if *all* mapped tnodes are virtual. If some are virtual and some are not, keep the loop but strip out the operations for the virtual tnodes (they have been inlined elsewhere).

4. **Partial loop cleanup**: add logic to `cleanup_virtual_llc` to filter out `Set`/`Zero_out` operations for virtual tnodes within a kept loop body, while preserving operations for non-virtual tnodes.

### Key files

- **`arrayjit/lib/low_level.ml`** -- all functions listed above: `visit_llc` (line 256), `check_and_store_virtual` (line 457), `inline_computation` (line 623), `virtual_llc` (line 775), `cleanup_virtual_llc` (line 860), `optimize_proc` (line 1388)
- **`arrayjit/lib/tnode.ml`** -- `known_non_virtual`, `update_memory_mode`, `is_complex` usage
- **`arrayjit/lib/low_level.mli`** -- public interface for `optimize_proc` and related types

### Risks

- **Circular dependencies**: two virtual tensors in the same loop that reference each other. `check_and_store_virtual` already detects escaping variables and recurrence, which should catch most cases. The `process_for` set in `virtual_llc` prevents infinite recursion.
- **Partial loop stripping correctness**: removing virtual-tnode operations from a mixed loop body while preserving iteration structure requires care. Operations may have ordering dependencies (e.g., a non-virtual tensor reads from a virtual one within the same loop iteration).
- **`is_complex` serves dual purpose**: it also reflects genuinely complex computations (via `is_complex_comp`), not just symbol sharing. The fix must only remove the symbol-sharing source of complexity, not the computation-complexity source.
