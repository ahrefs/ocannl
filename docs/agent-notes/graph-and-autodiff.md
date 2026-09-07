# Graph construction and autodiff

Forward/backward code ownership, fragment order, product-space gradients.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- A shared subtensor's forward code is embedded in its first-CONSTRUCTED consumer, and
  `Tensor.consume_forward_code` rejects consuming a root whose non-embedded reads are embedded
  in another live root. Consequences: construct (and compile) the consumer that should own a
  shared computation FIRST; with padding in play, the margin-demanding consumer (conv) must also
  compile before margin-blind ones, or the operand's layout locks unpadded. Sibling fragments
  are topologically ordered (owner-first forward, owner-last backward) — the old id-ascending
  order silently zeroed shared paths (regressions `forward_fragment_order.ml`,
  `backprop_fragment_order.ml`). A tensor's forward code is consumable ONCE: compiling one tensor
  several times (e.g. under different `?lowered_transform`s) means calling `Train.forward` once and
  reusing the comp it returned — the session records consumed roots, so the rejection says whether
  the code was already consumed, the tensor is a parameter, or a consumer owns its code
  (`consume_forward_code_reason.ml`).
- The `projections` handle an `op_asn` receives is a promise: `Tensor.op` creates the operation's
  shape update step only after `op_asn` returns, because the step's `neutral_elem` is immutable and
  is read off the assignments (`Shape.derive_projections` consumes it for the gh-504 clamped-window
  decision and for committing the margins' neutral, so a step with an unknown neutral element must
  never exist). Consequences: an `op_asn` must not force `projections.projections`, nor finalize
  inference by any other route (`to_dims`/`to_padding`): `Tensor.op` runs `op_asn` under
  `Shape.with_construction_window`, and `finish_inference` refuses while it is open, since the
  operation's constraints are not registered yet and its operands would close without them —
  rejected before the step is registered, at the finalizer (`neutral_elem_guard.ml`) — while `product_shape`
  and `*_pspace` intros stay usable there (the proxy shape is a function of the shape and logic
  alone); captured references are bound at registration, so `set_equal`/`set_scale` on them belong
  at the `%op` level, not in an `op_asn`. The rejection is a BUG REPORT, not a recovery API: shape
  inference is session-global and not transactional (gh-ocannl-903 enumerates the mutables — row
  substitutions, delayed refs, `unused_shapes`, the monotonic `Row` sets, graph roots, precision
  links), so the session is reset, and a per-mutable rollback at this seam is NOT to be grown: the
  review loop of lukstafi/ocannl-staging#598 tried, found another mutable every round, and was
  reverted. An operation that knows its accumulation up-front (`Tensor.raw_accum`) creates the step
  directly.
- Per-(result, reduced-position) facts in `%cd` grad code belong in a product-space intermediate
  (`*_pspace` suffix): its shape unifies with `Shape.product_space_shape` (result rows +
  contracted axes appended; fixed-index axes pinned to 1 — pinned projection means never a
  product axis even at extent > 1; ≤ 1 reduced-over row variable, else Shape_error), and its
  identity projection is `Indexing.prod_project_for`, which skips dim-1 axes and pairs the rest
  with product components by extent (first-fit — layout order need not match product order, since
  all accesses go through the same pure pairing); leftovers on either side raise at lowering. An operand-slot-shaped gate (`_rhs1`) is last-write-wins under
  overlapping windows — the gh-512 wrong-gradients bug; `tropical`/`einmax1` gates are now exact
  for stride < window and independent RHS2 indices, with an `=:|| eq (t1, t1)` validity mask for
  clamped windows whose output is genuinely -inf (executed oracles:
  `test/operations/overlapping_window_grads.ml`). Unary einsum specs with conv indices
  (`@^^ "o<+k => o"`, `++` alike) fail projection solving before any of this — pre-existing
  gh-515 — so overlap coverage goes through binary `@^+` with a zero kernel.
- Silent numeric divergence recipe: build a minimal numpy oracle reading the same safetensors,
  probe stage-by-stage (`Train.set_materialized` intermediates BEFORE `forward_once`, then
  `Context.get_values`), shrink to a tiny `NTDSL.init` repro, and read `build_files/*.cd` —
  use-before-def is visible at the .cd level.
