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
  `backprop_fragment_order.ml`). A `Taken` forward/backprop comp can be handed out repeatedly, including after `%cd`
  embedding: `consume_forward_code` and `consume_backprop_code` return the original comp without
  checking or modifying roots built since the first handout (gh-ocannl-955). This is equivalent to
  retaining the comp, including the caller's responsibility for execution order and gradient
  zeroing/seeding; compiling the replay a second time in ONE lineage is the lineage's to refuse
  (next bullet), so a replay compiles as a sibling. Parameters and embedded subtensors remain
  rejected (`consume_forward_code_reason.ml`, executed sibling replay in `consume_code_replay.ml`).
  The bare root removers are not exported: a route that drops a root without a consumer records
  its kind.
- Fragment ownership is checked per compilation lineage, not per tensor (gh-ocannl-1724). A
  fragment is `(Tnode.t * side)`: `(t.value, Fwd)` is `t`'s forward code, `(t.grad, Bprop)` the
  code propagating FROM `t.grad` (other fragments write that gradient too, so the written node is
  not the identity). `Tensor.op` fills `comp.owned` (own fragment plus embedded children's) and
  `comp.requires` (`(ti.grad, Bprop)` for each differentiable non-parameter child whose backprop it
  skipped because the child was no longer a backprop root), derived from the child's own `owned`,
  so a tensor built by `Tensor.term` (leaves and parameters: constant-empty `grad_asn`,
  `~terminal_backprop:true`) owns no backprop fragment and is never required — the maintainer's
  "exempt all terms". Zeroing, `init_params`, hand-written `%cd` assignments and the host-read
  recompute proxy own nothing; `embedded_nodes` stays an INITIALIZATION claim and is not ownership.
  `Context.compile` refuses a comp owning a fragment its frontier's `owners` already holds (keyed
  by node uid: a context must not keep every compiled node alive, which `moons_demo_variant`'s
  accessible-headers golden caught);
  `Context.run` refuses unmet `requires` against the running context's frontier when it descends
  from the routine's compile (same ledger, routine in `lineage_routines`), else the routine's own —
  never a sibling's (the ledger's `fragment_registry` only names one in the refusal). Traps: a
  required gradient is a `+=` (read-before-write), so with the gradient materialized — always under
  `Train.to_routine`, which materializes guessed outputs — the LINK fails first, and
  `Backends.Prior_context_lacks` carries the node to `Context`'s lineage hint; a test that needs
  the run-time check allocates the gradient first (a zeroing routine). Owner presence is not proof
  of a correct gradient: a skipped child's ZEROING is skipped too -- next bullet. A test that
  recompiles one program along a
  lineage to exercise lowering compiles an unowned copy (`Asgns.comp ~embedded asgns`). The
  `Train.*_once` functions mean "compile and run one time"; `forward_once` drops nothing, so a loss
  built after it trains through it (`lineage_fragments.ml`).
- A shared gradient's life cycle is checked per lineage too (gh-ocannl-1744): `Asgns.comp`
  carries `zeroes` (tensor-generated zeroing: `Tensor.op`'s `zero_grads`, `force_param_diff`) and
  `accumulates` (the operands' gradients `own_grad_asn` writes, root or not), and
  `check_shared_gradient_reset` beside `check_no_duplicate_fragments` refuses a second accumulator
  of a gradient unless neither zeroes it and every zeroing in the lineage precedes the first
  accumulation (`Z0; A1; Z1; A2` is refused), and refuses any contribution compiled after the
  routine owning `(g, Bprop)` (its propagation already ran). So `grad_update l1` and
  `grad_update l2` over a shared term or trunk cannot share a lineage; the supported spellings are
  one routine over the sum, or `zero_params_grads ~params` first plus `grad_update ~accum_steps:1`
  per objective (`filter_out_grad_zeroing` strips only `grads ∩ zeroes`, two-sided against the
  emitted `Fetch`es). Traps: siblings bind the ancestor's gradient buffer, so a sibling's value
  claim needs its own reset; a routine reading a term whose init another consumer embeds needs the
  value uploaded (`Context.set_values`), and a hand-built owner (`l1.grad =: 1; l1.backprop`)
  needs `l1.forward` for its constants (`shared_grad_refusal.ml`). A loss over a leaf whose first
  consumer is not compiled in the lineage (evaluated forward-only, since dropped, or a sibling's
  update) accumulates into a gradient it neither zeroes nor embeds, so no routine allocates it: the
  LINK fails before this check sees anything, and `Context.lacks_node_error` names the missing reset
  and `zero_params_grads` (`lacks_grad_reset_hint.ml`).
- Past the link, a gradient a routine accumulates into without zeroing is a RUN-history question
  (gh-ocannl-1781): the ledger's `accumulated` counts, per gradient buffer and per routine, the runs
  since the buffer's last reset (a run of a routine whose `zeroes` holds it, or
  `record_external_writes`, zero or not), and `run` refuses the run past the comp's
  `accumulation_runs` (default 1; `grad_update ~accum_steps:k` states k), naming any lineage zeroer
  that did not run. A static "no zeroer in the lineage" rule would miss the usual bug (a reset
  compiled and skipped) and refuse a `set_values` reset. Per routine, so `Z; A1; A2` cycles stay
  legal; the autotuner's `time_routine` passes `run ~timing:true` (its repeats count, are not
  refused), and its gh-1621 state restore puts back the history saved with the snapshot
  (`save_accumulation_history`), since restoring uploads are no reset; a hand-written `%cd` zeroing
  is no reset (`never_zeroed_grad.ml`).
- Within ONE comp, `owned` is a set, so a fragment carried twice merged unseen while `asgns` ran
  it twice: `Asgns.sequence` refuses members sharing a fragment, naming it (gh-ocannl-1746,
  `fragment_sequenced_twice.ml`). A comp built past `sequence` (a hand-made `Seq`) owes the same
  disjointness. The check found OCANNL's own duplicate: a `%cd` `op_asn` naming an operand tensor
  (`deinterleave_even`'s `id t1`) re-embedded the operand's forward beside `Tensor.op`'s `fwds`,
  so `Tensor.op` hides the root operands from the root map while `op_asn` runs. `%cd` builds a
  statement sequence in source order (gh-ocannl-1770: `ppx_cd`'s `source_ordered_list` let-binds
  every member but the last, since OCaml evaluates a list literal right to left), so the FIRST
  statement naming a root tensor embeds its forward code and later ones read it computed; an
  explicit `t.forward` belongs before the statements naming `t` — after one, it is refused as a
  second copy (`cd_source_order.ml`). Any new ppx site that builds a list of comps with effects
  (takes, embeddings) owes the same binding.
- The backprop-root map is weak and the forward-root map strong (gh-ocannl-1745). A backprop
  entry is an ephemeron keyed by the tensor's `consumption` cell (`Weak_root` in `tensor.ml`),
  live while any ownership-preserving record copy is reachable, so a graph dropped after its
  forward was handed out is reclaimed, checkpoints included. Forward roots stay strong:
  `consume_forward_code`'s conflict check is the only guard on forward reads (a lineage has no
  forward `requires`), and it must not depend on GC timing. So an unevaluated dropped graph stays
  pinned, as does a root captured by the last closure until the next closure. Traps: a stray
  reference to ANY record sharing the cell keeps the entry alive. A GC witness builds the graph
  in an `[@inline never]` function returning a `Weak` observer, and reads it with `Weak.check`
  after `Gc.full_major` (`dropped_graph_reclaimed.ml`).

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
  links), so a construction that failed after mutating taints the session and
  `Tensor.rewind_to_last_closure` discards everything open since the last closure (gh-ocannl-1301,
  `docs/shape_inference.md` "Session consistency"); a per-mutable rollback at this seam is NOT to be
  grown: the review loop of lukstafi/ocannl-staging#598 tried, found another mutable every round,
  and was reverted.
- Rewind restores the graph-root maps WHOLESALE from the checkpoint (the last closure's, or an
  explicit one, gh-ocannl-903) and resets the restored roots' handout markers — a `Taken` marker
  does not mean "handed to a caller": `%cd` embedding calls `take_forward_code` too
  (gh-ocannl-1301), but only on a root, so the roots' markers are all a window can set. External
  handouts (`consume_*_code`) are recorded as they happen in `handouts`, which a rewind subtracts
  from the saved roots and no rewind takes back; `with_unchanged_roots` restores it with the maps.
  A rewind discards exactly the nodes the frontend allocated since its checkpoint (gh-ocannl-1505;
  `Tensor.open_nodes` holds them): a new `Tn.create` site in
  `tensor.ml` must call `note_open_node`, or its node outlives the rewind as a live handle (an id or
  uid range would refuse nodes other code created in the window). `open_node_scan` (gh-ocannl-1579)
  enforces it over every `tensor/` module: the creators are derived from `tnode.ml` (the values
  minting a `fresh_uid`), and a creation's node must flow, through result positions of `let`
  bindings, into a variable passed to `note_open_node`; its header in
  `test/support/open_node_scan.ml` lists the forms it follows and what it refuses. A new `Context` entry point over a
  node runs `check_node_valid` (or, over a program, `frontend_gate`) before it forces anything: a
  discarded node's dims otherwise fail as an unrelated shape error. A new write a failed construction
  can leave behind must call `Row.note_mutation`, or a failure after it will not taint. An operation that knows its accumulation up-front (`Tensor.raw_accum`) creates the step
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
