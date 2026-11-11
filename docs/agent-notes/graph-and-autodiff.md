# Graph construction and autodiff

Forward/backward code ownership, fragment order, product-space gradients.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- A shared subtensor's forward code is embedded in its first-CONSTRUCTED consumer, and
  `Tensor.consume_forward_code` rejects consuming a root whose non-embedded reads are embedded
  in another unconsumed root, held or dropped (owner cells, below). Consequences: construct (and
  compile) the consumer that should own a shared computation FIRST; with padding in play, the margin-demanding consumer (conv) must also
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
  recompute proxy own nothing -- which is also what tells a deliberate replacement from a forward
  recomputing a stepped node (`Asgns.persists`, gh-ocannl-2027, training-and-performance.md); `embedded_nodes` stays an INITIALIZATION claim and is not ownership.
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
  consumer is not compiled in the lineage (evaluated forward-only and held, or a sibling's update)
  accumulates into a gradient it neither zeroes nor embeds, so no routine allocates it: the LINK
  fails before this check sees anything, and `Context.lacks_node_error` names the missing reset and
  `zero_params_grads`, plus any routine of the root compiled outside the lineage that zeroes it
  (the ledger's `zeroing_registry`, gh-ocannl-1797; `lacks_grad_reset_hint.ml`).
- The "first-constructed consumer" is the first that can still run (gh-ocannl-1804): a backprop
  root reclaimed with its backprop unconsumed gives back the LEAVES it absorbed, transitively, so
  the next-constructed consumer carries their zeroing. Each root's `Weak_root` entry keeps the
  leaves' own entries in a shared rope, and `Weak_root.prune` returns them. Only a dead entry still
  in the map gives back: a consumer handed out (its routine still owns the zeroing), or embedded by
  a live parent, has left the map, and a held record copy keeps the entry alive. Computed tensors
  stay unrestored, forward ownership and owner cells are untouched, and a tensor built earlier is
  not retrofitted. `Tensor.op` decides at construction, and no closure need separate a consumer's
  death from the next loss, so the prune runs in `op` too, for a non-root leaf operand. Traps: `op`
  settles its operands' rootness ONCE into `bck_roots` before any read, because a GC between two
  reads (or a nested op under `grad_asn`) could embed a gradient whose zeroing the first skipped; a
  restore filters `handouts.bprop_out` BEFORE pruning, through the entry's `grad_uid`, since a dead
  owner's code can outlive it in a routine. The outcome is the GC's to time, on purpose: a consumer
  dropped but not yet collected refuses as a held one does, and the next loss then links only after
  collection. Both outcomes are safe, a correct gradient or the loud refusal, whose text says so;
  `Train.zero_params_grads` works in both, `Gc.full_major ()` after the drop settles it, and summed
  losses over one reset spell `grad_update ~accum_steps:1`, since an `A1; A2` with neither zeroing
  is refused once the first takes the zeroing back. So a test of "this loss does not zero" must HOLD
  its consumer past the loss's construction (native code keeps no dead local alive), and a test of
  the hand-back must observe the reclamation with `Gc_witness` first (`reclaimed_consumer_zeroing.ml`).
- `debug_gc_backprop_roots=true` forces a full major collection in `Tensor.op` immediately before
  the hand-back scan, only when an operand is a non-root leaf (gh-ocannl-2062). The default-off
  switch settles reclamation without changing ownership rules. The `cc-gc-backprop` sweep unit
  runs all of `@runtest` under it; the quick CPU alias `@test/operations/gc-stress-backprop`
  pins its effect without an explicit collection in the fixture. Set the key externally to stress
  other aliases or backends. Its boundary is the hand-back gate, not every weak-root read.
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
  statement sequence in source order (gh-ocannl-1770: `ppx_cd`'s `source_ordered_list`, through
  `Ppx_shared.in_source_order`, let-binds every member that can have an effect but the last,
  since OCaml evaluates a list literal right to left), so the FIRST
  statement naming a root tensor embeds its forward code and later ones read it computed; an
  explicit `t.forward` belongs before the statements naming `t` — after one, it is refused as a
  second copy (`cd_source_order.ml`). Any new ppx site that builds a list of comps with effects
  (takes, embeddings) owes the same binding.
- Both root maps are weak: backprop since gh-ocannl-1745, forward since gh-ocannl-1796. An entry
  is an ephemeron keyed by the tensor's `consumption` cell (`Weak_root` in `tensor.ml`), live
  while any ownership-preserving record copy is reachable, so a dropped graph is reclaimed,
  evaluated or not, checkpoints and the last closure included; a handed-out comp a caller keeps
  pins its tensor through the projections. `consume_forward_code`'s
  conflicting-roots check does not read the map: it reads OWNER CELLS (`owner_cell`, in
  `consumption.fwd_owner`), a GC-independent shadow of "is a forward root" with one owner edge per
  embedding (a DAG, not chains: code replayed into two roots is owned by both, and neither owns the
  other's result). A weak `Owner_cells` table maps each value node, and each node an op's own
  `op_asn` embeds, to its cell; `Tensor.op` adds the edges at registration. A root dropped
  unconsumed keeps `is_root`, so its readers stay refused whatever the GC did. Map entries hold
  their cells strongly so restores (`restore_root_cells`) reset a reclaimed root's cell like a
  live one's; `with_unchanged_roots` defers forward pruning and owner-list compaction (by
  `closure_epoch`) so its restore sees every root its body made. Traps: a new source of a root's
  `embedded_nodes` must add owner edges, or it silently stops refusing: the
  `debug_check_forward_root_owners` check, on across the suite in `test/config`, fails any
  `consume_forward_code` whose scan of the live roots finds a root the cells miss (a subset, not
  equality: the scan loses what the GC reclaimed); the refusal names roots by `label_name`. A stray
  reference to ANY record sharing the
  cell keeps the entry alive. A GC witness builds the graph in an `[@inline never]` function
  returning a `Weak` observer, and reads it with `Weak.check` after `Gc.full_major`
  (`dropped_graph_reclaimed.ml`).

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

- Top-down precision links are guarded by shape closure, not by a node's `dims` readiness
  (gh-ocannl-2026). The precision cell owns the forced-read refusal and the `Specified` no-op.
  `dims_top_down_prec` supplies known layout on open tensor record copies to separate these
  states: ordinary dataless frontend dims close inference when forced, while data-backed leaves
  already have specified precision. Wrapped leaves are born with forced dims on an open shape;
  they now reach the `Specified` no-op before the forced-read check, pinned by the read-first
  `wrap` rows in `data_leaf_top_down_prec`. The algorithm is in `docs/precision_inference.md`.

- Embedding a seed singleton does not initialize it (gh-ocannl-2023). When `Tensor.init_params`
  adds the singleton whose code another graph absorbed, it also prepends that seed's own
  `Constant_bits` fetch. This covers independent models and unused branches without requiring
  callers to initialize a library-owned node. Replaying the literal does not reset a stream:
  `uniform` derives its counter from node ids and offsets, and `uniform_at` adds an explicit
  counter input. `embedded_read_only` pins draws against explicit seed 42, distinguishes seed 0,
  and repeats parameter initialization. Other unwritten embedded read-only nodes are still
  refused by `Context.run`; the seed is accepted because its initializer now writes it, not by
  an exemption. A seed that is no longer the singleton -- captured inside a
  `with_saved_random_seed` scope since restored, or replaced by `set_random_seed` -- is repaired
  the same way (gh-ocannl-2119): `set_random_seed` registers every seed literal weakly with its
  fetch (`seed_literals`), and the outer `init_params` prepends the fetch of each registered seed
  its program reads (`collect_nodes_guess_output`'s first set) and does not write
  (`collect_written`), once over the whole program. The rule, not a refusal, because the literal is
  immutable whichever singleton is current; `embedded_read_only`'s captured-seed cases pin the
  draws against a caller-seeded reference and a seed-8 control across nested, unwound, replaced
  and same-integer variants (the registry keys on the seed node, which the kept graph holds, so the
  owner's liveness does not enter; a `Gc_witness` probe found the dropped owner still reachable
  before initialization, so it is not a variant). Forward code follows the same rule
  (gh-ocannl-2157): `with_seed_fetches` is the one transformer, and every forward handout
  (`consume_forward_code`, both branches, and `take_forward_code`, `%cd`'s embedding) returns it
  applied, memoized per tensor record when it changed anything (`forward_handouts`, held weakly by
  the record, never by its node, which outlives a dropped tensor), because repeated handouts are pinned
  `phys_equal`. `forward_reads` reads the repaired handout, so a
  seed is never a conflicting read: an unused draw that absorbed the seed no longer refuses its
  readers. The invariant: a routine reading a seed writes it, whatever context it links in. It
  re-runs the fetch in a lineage that already wrote it, and a `%cd` block embedding the owner and a
  reader carries two fetches of one literal: both harmless, pinned by `forward_seed_fetch` against
  a reference that reads the seed from its owner's lineage. A raw `t.forward` given to
  `Context.compile` bypasses the handout and gets no fetch.

- Parameter initialization order follows the program (gh-ocannl-2158): `init_params` orders
  fragments by dependency, with first forward-statement access and tensor operand order breaking
  ties. The shared `topo_layers` helper also orders forward/backprop fragments. A seed fetch can
  live inside the first-created parameter, so initialization hoists registered seed literals before
  ordering the draws; otherwise that accidental dependency would keep creation order in the
  parameter pool offsets. No node or counter is reminted: `build_order_programs` pins fixed-graph
  random draws bit for bit against separately initialized parameters, executed shared-initializer
  dependencies (including copying slices), and held-parameter offsets across reversed conv/transformer builds (including
  `buffer_aliasing`). Reversing graph construction still changes node-keyed random counters;
  cross-build loss comparisons therefore keep the fixture's label-keyed parameter uploads.
  The dependency query adds slice-parent reads explicitly: `collect_nodes_guess_output` instead
  classifies those parents as materialization outputs, so its first set alone misses the edge.
