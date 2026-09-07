# Lowering: analysis, digests and caches

What the analysis pass establishes about a routine, how the two digests over lowered code stay
honest, and the lowering-time seams: traced store, affine paths, index precision, slice aliases.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- **Which operands of an op actually evaluate is `Ops.binop_conditionality` /
  `Ops.ternop_conditionality`, and nowhere else** (gh-ocannl-582). Both matches are exhaustive, so
  adding an op forces the decision; the consumers are `Low_level.affine_accesses` (a discarded
  projection operand makes no access), `Cost_model.analyze` (upper), `Cost_model.floor_flops` plus
  its `floor_uncertainty` pre-pass (floor), and `C_syntax.pp_scalar` / `debug_float` (a projection
  emits its selected operand alone). Before the classifier existed the same case analysis was
  restated in each, and one phase-3 review re-aligned them three times running — `Where`, then
  `And`/`Or` and the projections, then `Relu_gate`/`Satur01_gate`. The *renderers* do not consult
  it; they are checked against it by `C_syntax.operand_conditionality_violations`, run once per
  `C_syntax` functor application (i.e. per compile, on whatever hardware the backend has — the GPU
  syntax configs are sealed inside their `Impl` and unreachable from tests), which renders every
  (precision, operator) pair over placeholders and looks for the only C-family constructs that skip
  an operand: `?:`, `&&`, `||`. This is what forbids spelling `Where` as MSL's `select` or `Max` as
  `(a > b ? a : b)`. The operator sweep (shared with `op_syntax_idents`) is `Ops`' derived
  enumeration — the four operator types carry `[@@deriving enumerate]` — so a new operator cannot
  escape either check by being left off a hand-maintained list. `Ops.prec` cannot be derived that
  way (its constructors carry the phantom-typed `precision` witness), so `C_syntax.all_precs` keeps
  a hand list next to an exhaustive match that turns a new precision into a build error.
- The upper cost walk charges every operand that is *rendered*, which is more than can *execute*:
  a `Where` arm's (or a gated operand's) `Local_scope` renders as statements hoisted OUT of the
  conditional expression — `C_syntax.pp_scalar` returns its definitions separately — so both arms'
  scope bodies really do run. Only an operand no renderer emits at all (a projection's discarded
  one) may be dropped from the upper bound. The floor's `Int.min` is unaffected: hoisting only
  loosens a lower bound.
- **A materialized constant literal's initialization is data, not code** (gh-ocannl-633): at or
  below `limit_constant_fill_size` a `Tensor.ndarray` literal carries an in-kernel `Constant_fill`
  fetch — its inlining recipe — AND registers its values in `Host_inits`. When a lineage's
  placement materializes the node, `Low_level.hosted_constant_inits_to_link_time` (in
  `specialize_proc`, right after `simplify_llc`) deletes the in-kernel init writes and flips the
  traced facts to read-only input, so the node self-initializes at link time exactly like an
  above-threshold (`Reshape`-backed) literal. Consequences: schedule legality does not depend on
  operand literal size (the straight-line init writes used to fail `validate_parallel`'s coverage
  rule beside hardware-annotated loops); a fresh context can link a routine reading a constant
  whose fetch an earlier routine consumed (`verify_prior_context` exempts `Host_inits` members);
  and `Stage ~hoisted:true` reaches small constants (`Schedule.hoistable_constant`). Eligibility
  is `Tn.known_host_constant` — the persistent marker, not the `Effectively_constant` intent, so
  an explicitly `Train.set_materialized` literal still converts — and any literal-constant write
  form is droppable under that contract: unrolled fixed-index `Set`s (`Constant_fill`), loop-borne
  `Set`s (broadcast `Constant`), and whole-node `Zero_out` (`Constant 0.`, unreachable by the
  `limit_constant_fill_size=0` escape since 1-element literals never consult the limit). Bail-outs
  that KEEP the in-kernel init: padded constants (their init includes padding-region loops),
  `Local`-placed constants (scratch is fresh per launch, uploads cannot reach it), any
  non-constant write form, and constants the routine never READS — a write-only literal root (the
  `Train.forward_once`-then-print pattern) is an explicit "compute this constant into the
  context"; converting it would empty the routine and push observation onto print-proxy fallbacks.
  Test: `test/operations/hosted_constant_fill`.
- Both digests over lowered code — the analysis cache's key (`Low_level.analysis_digest`) and the
  schedule cache's canonical identity (`Schedule_cache.canonicalize`) — share ONE walk,
  `Low_level.Canonical_render.emit` (gh-ocannl-563). Render a newly added `Low_level.t` /
  `scalar_t` construct there and nowhere else (the matches are exhaustive, so the build breaks);
  a new digest-relevant *fact* is a `Canonical_render.policy` field if it is an identity choice, or
  a consumer's own preamble/companion section if only one of them consults it. The walk owns what
  the two agree on: loop-binder tokens `b<n>`, first-occurrence local-scope alpha, comment
  skipping, `mark_incomplete` on opaque statements. Golden + seam test:
  `test/operations/canonical_render.ml`.
- Why a bespoke renderer rather than `to_doc` / `sexp_of_t`: (1) the digests must be
  ALPHA-INVARIANT — `Indexing.symbol` and `scope_id` are global counters, so sibling lowerings of
  one routine share no symbol numbers and any structural print misses 100% of the time; (2)
  `to_doc`'s node names are NOT CONTEXT-FREE — `get_ident_within_code` pre-passes the whole code
  array to find labels claimed by more than one uid and only disambiguates those, so a node prints
  as bare `x` when alone, which COLLIDES two different routines rendered separately (a wrong hit,
  and for the analysis cache a correctness failure) and makes a fragment's digest shift when a
  sibling fragment changes (per-segment schedule matching needs the opposite); it also mutates
  (`Tn.update_code_name`) and is layout- and config-dependent (`PPrint` width,
  `ll_ident_style`/`output_prec_in_ll_files`). (3) `Low_level.t` derives no `compare`/`hash`
  (`Staged_compilation` holds a closure), and the schedule cache's key becomes an on-disk FILENAME
  (`Schedule_cache.cache_key`/`cache_file`) — so a string digest, not a structural key.
- The `Low_level` analysis cache (gh-ocannl-560) makes sibling candidate compiles share one
  `analyze_proc` result keyed by that digest of the raw lowered code. Its identity policy is the
  OPPOSITE of `Schedule_cache.canonicalize`'s: tensor nodes and static symbols enter by identity
  (`Tn.uid`; symbol ident + the mutable `static_range`/`used_as_extent` facts) because a hit
  reuses the stored code verbatim, while `canonicalize` alpha-renames everything so schedules
  replay across sessions. Anything the analysis consults beyond the code must enter the key —
  `inline_complex_computations` does, since the rmw exemption changes what the coverage /
  multiplicity queries count. Two traps: (1) caches
  that retain lowered code keep tensor nodes (and, via pool finalizers, buffers) alive — register
  a clearer in `Tnode.before_accessibility_snapshot`, or `print_accessible_headers` goldens grow
  phantom "accessible" nodes (this is how the cache was caught); (2) on a hit, still re-run
  `pin_device_written_bounds` — its raising writer-after-settled-reader guard must fire regardless
  of caching.
- **The traced store is the routine's node registry, and it is RECONCILED with the FINAL
  optimized code** (gh-ocannl-610): kernel parameters (`C_syntax.compile_proc`), context
  allocation (`Backends.allocate_delta`) and the routine interface
  (`Low_level.input_and_output_nodes`) all enumerate `optimized.traced_store`, which
  `analyze_proc` builds from the RAW code — cross-routine inlining (a splice of a computation an
  earlier routine committed `Virtual`) makes the two diverge in both directions.
  `specialize_proc`'s `reconcile_traced_store` walks the final llc in program order and: gives
  spliced-in nodes fresh `read_only` entries (codegen otherwise emits an undeclared identifier);
  flips an already-traced node to `read_before_write` when a spliced read lands before the
  routine's own first write, and re-judges reads AFTER a write with the same per-cell machinery
  the raw pipeline uses (`reads_covered_query` over the final code's affine accesses, built
  lazily) — syntactic priority is not coverage: a write of some cells or under an `If` covers
  nothing it did not touch; prunes read-only entries the final code never touches (phantom
  inputs of deferral-only routines) while KEEPING entries that record a raw write even when
  unaccessed — out-of-contract probes (gh-ocannl-584 scope writes, `affine_extraction.ml`) check
  raw decision-level facts, so the prune must stay no wider than the phantom-input genre; and
  drops a raw-declared merge node whose read was deferred away (linking would otherwise demand a
  transfer never consumed). Merge splices are rejected at CONSUMPTION time, not post-hoc:
  `virtual_llc` snapshots which inherited computations read a merge buffer at entry (before the
  walk stores this routine's own), and `inline_computation` raises on consuming one — the final
  code cannot distinguish a legitimate same-routine inlining of the declared merge read from a
  cross-routine splice whose consumer declares the SAME source, which would silently rebind the
  read to the consumer's transfer; `reconcile_traced_store` keeps a mismatch check as backstop.
  The walk observes the raw analysis' conventions or it re-diverges (review rounds 3-6):
  dead-loop bodies register (renderers emit them, so their identifiers need parameters) but
  neither supply coverage (`written_seen`) nor demand it, and the coverage query filters through
  `drop_dead_loop_accesses` like the raw side; `Binop` dispatches through
  `Ops.binop_conditionality` in both the reconcile walk and the merge-taint scan — a projection's
  discarded operand is never rendered, so its reads are not parameters and its merge read must
  not taint; the taint scan is `~self`-filtered like `inline_computation`'s own setter filter, or
  a shared-loop sibling's merge read rejects valid sharing. The STRICT coverage verdicts —
  guarded writes filtered (never definite), rmw exemption not counted as coverage (a
  same-position read is a genuine RMW), `zeroed_out` counted as written — apply PER NODE, to
  exactly the nodes read inside INLINED bodies (`virtual_llc` records them at each
  `inline_computation` splice and returns the set): splicing is what moves reads to positions the
  raw analysis never judged. Raw-positioned reads keep the raw GUARDS-TAKEN contract, which
  deliberately classifies patterns initialized by an earlier routine of the program —
  routine-wide guard strictness broke real flows across the suite, and a has-local-assignment
  provenance test for "inherited" missed consumption through an update of an inherited virtual
  (rounds 6-7). The rmw part is split more finely (gh-ocannl-618):
  `reads_covered_query` returns a three-way verdict, and exemption-dependent coverage
  (`` `Covered_rmw_exempt ``) counts as covered for the tracer-mirroring placement heuristics
  (the visit cap) but as uncovered for the `read_before_write` interface classification of a
  node that ends up owning a buffer — a copy-position read after a partial write, or an
  accumulation with no preceding definite initialization, consumes entry values raw or spliced
  alike (`splice_semantics.ml` phase 1; routine-complete lowered flows are unaffected because
  lowering emits the initialization first). The strict classification runs in
  `reconcile_traced_store`, over the SETTLED placements, and promotes the flipped node
  `On_device` (an entry-consuming node must not resolve to `Local` scratch — round 3) — not in
  `decide_placements`, where two wrong timings lurk: judging undecided nodes strictly destroys the virtualizer's partial-write
  producers (an injective scatter emits no neutral init; inlining prepends the init fallback —
  `affine_lowering.ml` AC6 broke under that reading), while judging only already-decided nodes
  misses candidates flipped non-virtual AFTER the decider (the fan-in guard, a
  `check_and_store_virtual` legality rejection such as a guarded RMW — the PR's round-1 P1).
  A node that stays virtual is exempt by construction: it has no interface. `from_prior_context` (both `Backends.compile` and `from_prior_context_batch`)
  reconciles in BOTH directions: the raw-assignments set is filtered by the reconciled traced
  store (raw over-approximates the residual schedule — a deferral-only routine must link on a
  fresh context) and, for routines carrying an assignments program, unioned with the reconciled
  interface's inputs the raw asgns never MENTION (raw also UNDER-approximates: a consumer whose
  asgns read only the virtual node would otherwise zero-fill its spliced leaves silently). Both
  bounds on the union are load-bearing: mentioned nodes keep `context_nodes`' curated exclusions
  (init comps' random-seed/threefry nodes are mentioned yet deliberately not demanded — the
  unbounded union broke `Train.init_params` across the suite), and hand-built `?prelowered`
  routines (empty comp) are exempt entirely — their inputs arrive via `Context.set_values` after
  linking, the ll_test seed-then-run pattern. Reconcile-FLIPPED read-before-write nodes
  (`optimized.spliced_rbw`) override the mention filter — a consumer that overwrites a spliced
  leaf mentions it only as a write, yet the splice needs its entry value; the raw
  `read_before_write` flag cannot serve as the key, since `decide_placements` also sets it on
  every pure input (uncovered reads), and demanding those broke ndarray-literal flows. The merge SOURCE never gets an ordinary traced
  entry (the merge buffer is the parameter; a source entry would double the transfer buffer's
  allocation).
  Post-finalization lineage tests use `Ll_test.optimize_in` over one optimize context, then
  `Ll_test.link_finalized` on the producer: real backend codegen resolves retained
  `Never_virtual` entries, and `Ll_test.known_local` proves the follow-on guard saw `Local` rather
  than the broader `known_not_materialized` class (gh-ocannl-631).
  Corollary (gh-ocannl-611): a routine whose every statement virtualizes away is LEGAL —
  cleanup's top-level elision degenerates to `Noop`, with an EMPTY interface — its stored
  computations persist in the lineage for later consumers, so "compile a deferral-only routine"
  is a supported incremental-compilation move. Deferred computations DO observe inputs mutated
  between deferral and consumption — recompute-at-read is the decided semantics (gh-ocannl-617,
  option 1): a virtual node is a named computation, not a snapshot, and at the arrayjit level
  the recompute-vs-materialize semantics is deliberately not fixed — the memory-mode intent and
  the choice of routine boundaries are the user's knobs (see "Recompute-at-read" in
  docs/lowering_and_inlining.md; `splice_semantics.ml` phase 2 pins both readings of one
  program text). The acceptance pins are `test/operations/virtual_chain_fanin.ml` phases 3–5.
- **A knob read after lowering cannot reach a digest over lowered code** — it must be carried by a
  cache-key component or the cache replays across regimes (gh-ocannl-568: 5.9x). So every config
  key is classified in `Utils.config_key_classification` as code-borne / `Keyed <component>` /
  search-shaping / execution-neutral, with the reason, and `test/operations/digest_completeness`
  fails on an unclassified key, on a claimed component that does not exist in
  `Schedule_cache.key_components` (that list DRIVES `cache_key`, so it cannot go stale), and on a
  key read in a codegen-stage module yet classified code-borne. When adding a config key, classify
  it; when adding a backend knob consulted at codegen, put it in the backend's
  `hardware_limits.codegen_tag` — `cache_key` takes the whole limits record precisely so a new
  component reaches every call site. `digest_identity_flips` calibrates one representative per
  class against a real compile; when picking a code-borne representative note that many optimizer
  keys (`virtualize_max_visits` and its neighbors) are read ONCE into `Low_level.virtualize_settings`
  at module init, so poking `Utils.config_file_args` at runtime does not move them. Caches with an
  identity of their own still state their own extra inputs (the analysis cache's
  `inline_complex_computations`, the cc probe cache's compiler/arch/simd settings).
- `Affine.access.a_path` components are TYPED (`Affine.path_comp`, gh-ocannl-561): `Stmt` indices
  interleaved with `Cond`/`Body` (`If`) and `Rhs`/`Write` (`Set` family), constructor order =
  execution order, so lexicographic comparison is program order within a statement too. Every
  access path ends in `Cond`/`Rhs`/`Write` and nothing extends past `Write`. Consumers must not
  compare bare paths for "same statement" — use `Affine.same_statement` (agreement above the
  final component; the path-level twin of `a_stmt_write` subordination) — and must take top-level
  statement identity via `Affine.stmt_head`, never `List.hd` (a single-statement routine's paths
  start with a marker, not a `Stmt`). History: with bare positions an `If` condition's read
  aliased its guarded body's write (gh-554 round 3), and `read_covered_before` needed a
  prefix-exclusion hack for enclosing writes vs their inlined `Local_scope` bodies — both
  unrepresentable now. Each `Local_scope` occurrence additionally extends the path with an `Arg`
  evaluation position (per-statement counter), and `path_before` deliberately does NOT order
  across sibling `Arg`s: two scope bodies inlined into one statement must neither interleave
  their interior components (a `Seq`-bodied sibling's `Stmt` sorts before a bare-bodied one's
  `Rhs` — a later operand's write would pose as prior to an earlier operand's read; Codex P1 on
  staging#297) nor claim cross-operand evaluation order at all (it would silently depend on codegen's
  scope emission order).
- **`Ir.Affine` owns the peel-guard rule** (gh-ocannl-722), which is the pattern to follow when a
  legality question starts growing clauses somewhere else: `Affine.separates` is `pair_conflict`
  applied to one access taken twice (the instance-vs-instance form — two instances of the same
  statement, iterating `concurrent` symbols independently, can share a cell only if they agree on
  every symbol of `syms`), and `Affine.peel_guard` is the symbol-set classification that decides
  when to ask it, with `Affine.within_box` — the interval companion of `covers_box` — answering the
  access-validity half that separation does not. `Low_level.peel_accum_nest` is the consumer; `Low_level.loop_bounds` is the box
  environment both take (do not add a second walker for it — round 7 of gh-693 did, and gh-722
  removed the duplicate). Soundness direction, as everywhere in the engine: a proven "separated" is
  proven, and anything it cannot interpret declines.
- **The peel's DECISION is carried, not re-derived from the emitted kernel** (gh-ocannl-733). Two
  nests differing only in whether the accumulated cell mentions the enclosing index peel a different
  number of levels under a different guard verdict and emit the SAME localized kernel — one scope,
  one closing store — so classifying emitted code cannot say which path ran, and a test named after
  one of them stays green over the other. `Low_level.peel_accum_nest ~report` reports, once per
  call, how many levels it peeled, which verdict each peeled guard earned (`Guard_confined` /
  `Guard_lane_private`) and the refusal where there was one (`Refused_cell_varies` vs
  `Refused_cell_shared` vs `Refused_not_a_nest` vs `Refused_dead_level` vs `Refused_guard_fixed`);
  `C_syntax` accumulates that per routine and `Context.routine.peel` carries the summary, exactly as
  `routine.mma` carries the `Tile_mma` census. Read that field — do NOT bracket
  `peel_census_enabled` yourself (`C_syntax.with_peel_census` nests additively, and
  `Context.compile` calls it). Three things a reader of the census must know: its `levels` counts
  the rendered level too (the peel is asked of that level's BODY, so it reports one fewer); only
  sites carrying a CELL self-recurrence are censused (`Low_level.has_accumulating_cell`, NOT the
  conservative `has_accumulation`, which counts every `Local_scope` and so recorded an ordinary
  virtualized computation as a declined reduction site), with collection suspended while a localized
  rendering re-renders the peeled levels inside the scope it just minted; and a routine whose form
  is localized but whose census localized NOTHING had its scope minted upstream — by a `Schedule`
  mint (`Unroll ~materialize`, `Partition`, `Pad`) or by virtualization — which is a different fact
  about the kernel than "codegen localized it". The `reduction_forms` member table declares which of
  the four (`Peeled` / `Minted_upstream` / `No_localization` / `Never_a_site` — the last being
  BOTH virtualization members, whose accumulator lives in the inlined scope rather than in a cell, so
  there is no memory recurrence and no site at all) each member claims, and
  `test/operations/peel_census.ml` pins the instrument itself, including that a REFUSING report
  never claims an admitted lane-private guard: separation is settled at the base, so a refusal that
  never reached one leaves that guard `Guard_lane_private_unresolved`.
- `Ir.Ops.index_prec ()` is SIGNED (int32; int64 under `large_models`): negative index
  intermediates are well-defined; emit guards in natural signed form. Guard shapes are
  canonicalized to ONE shape per role: upper bounds are strict `Cmplt` (`idx < bound`, the natural
  operator), lower bounds are direct `Cmple` (`0 <= idx`). Construct new guards in exactly these
  shapes — recognizers match structurally (`schedule.ml`'s Tensorize mask parser and
  breakpoints-from-guards affine view, `c_syntax.ml`'s launch-guard strip), so an off-canon
  encoding silently stops contributing breakpoints or gets rejected. When adding a comparison
  operator to a guard, give every such recognizer an arm for it. Per-node element counts must fit
  int32 unless `large_models`; launch params are bind-validated.

- **An ordinary assignment never puts two DISTINCT tensor nodes under one `For_loop`** — each gets its
  own `Low_level.loop_over_dims` (`Assignments.to_low_level`), and a `Block` concat produces several
  loops that all write the SAME node. The exception is its mirror, `Rev_sides` (the `++^` gradient,
  which scatters one RHS into the component gradients): its sibling segment loops each write a
  different member of `lhses`, so an ENCLOSING product level's iterator — which indexes all of
  them — owns several tnodes, and `track_symbol` records them all. Shared-loop behaviour
  (`reverse_node_map : Symbol.t -> Tnode.t list`, gh-ocannl-134) is therefore reachable from the
  DSL after all; `test/operations/concat_loop_symbols.ml` pins that a batched concat gradient still
  produces it. The sibling segment loops themselves no longer share a binder: lowering mints one
  loop symbol per SEGMENT (gh-ocannl-765), because a shared binder is misread by every flat
  symbol-keyed scanner — `def_loop_ranges` keeps the last segment's width, `affine_accesses`
  collects two ranges for one symbol, and the canonical render calls the second binder shadowed,
  which declines the routine for both digest caches. `test/operations/test_block_tensor.ml`'s
  gradient fixtures are the regression coverage to keep in mind when changing concat lowering or
  the virtualizer's symbol ownership, alongside the hand-built
  `test/operations/virtual_shared_loop.ml` (`ll_test`), which is still the way to reach shapes the
  DSL does not emit.
- **The product-space loop nest is walked in ONE place**, `to_low_level`'s `with_product_loops`
  (gh-ocannl-764): it emits a loop per segment of each `Indexing.components` entry, applies the
  gh-490 extent guards, and calls back once per block (per choice of concat segments) with that
  block's substitution (`block_subst`), the fresh symbols' loop widths (the gh-504 clamp bounds)
  and an `is_allowed` selector over the block buffers. `loop_accum` and `loop_accum_rev` are then
  only their own role assignments. `Set_vec_unop` reuses `block_subst` but keeps its own nest, and
  its four divergences are commented at the seam rather than left to drift: the nest is split for
  the tail peel, `Concat` is rejected rather than resolved, `Empty_block` is unreachable there, and
  neither the extent guard nor the clamp is wired in. A bound gh-490 extent is explicitly rejected
  before the nest (gh-ocannl-817): packed uniform's round-up `Total_elems` constraint already keeps
  the shape out of derived projections, and the lowering refusal keeps a future path from turning
  that current solver limitation into an unguarded wrong-write. Projection metadata retains a
  maximum-one symbolic axis without an iterator, so runtime zero cannot escape the refusal through
  `Fixed_idx 0`. An unbound extent retains the standing maximum-shape semantics; padded-window
  clamping remains unreachable by construction.
- **A gh-490 extent guard is per-ITERATOR, so an axis with no product iterator gets none**, and
  `Accum_op` reaches that shape from ordinary `%op` code: a maximum-one symbolic axis has an empty
  product space, both sides indexed at `Fixed_idx 0`. Reducing over one bound to zero returned the
  operand where the empty sum is the accumulation's neutral element -- a wrong value in an
  IN-extent output cell, not merely a write past the logical shape (`test/operations/
  accum_max_one_extent.ml` executes both, against a maximum-two twin that guards correctly).
  `with_product_loops` now refuses a bound extent whose iterator is absent, for both the `Block` and
  `Rev_sides` roles (gh-ocannl-878). An enclosing `0 < extent` guard is NOT the local repair it
  looks like: with no iterator the assignment is surjective and injective, so
  `can_skip_accumulation` drops the neutral-element init, and skipping the write would leave the
  target holding whatever preceded it rather than zero -- soundness would have to couple the guard
  to the accumulation-elision decision, for a symbol that can only take the values 0 and 1.
- A lowering-time reader of `Indexing.variable_ref` must force a dims lazy (or otherwise finish
  inference) FIRST: forcing dims is what runs `Shape.finish_inference` and fills row-var-bound refs
  (`..d..` captures such as layer norm's `/. dim d`). "Inference is already forced by now" is not an
  invariant at the top of `to_low_level` — in a deep model the first statement lowered can be exactly
  that `Fetch Embed_dim`, which then raises "no solved dimension" (gh-ocannl-490; fast tests miss it
  because their row vars solve eagerly at stage 1 during graph construction).
- A `Fetch.Slice` (`@|`) alias is not a view of any shape-compatible parent. The materializing copy
  loop it replaces goes through scalar precision CONVERSION, so a slice legitimately has a different
  precision from its parent (`primitive_ops` slices a float `x_flat` into a uint4x32 `x`), and shared
  storage cannot convert — `Tnode.alias_of` eligibility requires `Ops.equal_prec` alongside the
  leading-axis rank drop and an unpadded, materializable parent. Host access of an alias view raises:
  read and write the parent, since `from_host` would otherwise allocate a detached buffer and break
  write-through. The alias never enters `ctx_buffers`, so finalization and pool resolution never see it.
- `Ops.promote_prec` is ordered `Uint4x32_prec` first, then the floats widest-first, then the
  integers: the RNG state precision dominates EVERYTHING (a join of uint4x32 with double is
  uint4x32, which is what keeps a threefry chain from being widened by a float consumer), and
  below it any float dominates any integer — fp8 over int64 included — so a precision-inference
  join through a float operand destroys integrality. Integer id chains (class
  ids, gather indices) must be pinned rather than inferred; `Tn.update_infer_prec` under a
  `not (Lazy.is_val prec)` guard is the threefry/one-hot precedent, and the gather guard's precision
  flavors (unsigned/signed/float) branch on the ids' storage precision, not on `index_prec`.
- `Indexing.Concat` axis indices exist only ABOVE lowering: the loop nest iterates a concatenation's
  segments one at a time, so what reaches an access index is the segment's own iterator. Post-lowering
  code (`schedule.ml`, `c_syntax.ml`) therefore never sees one, and a `| Concat _ -> false` arm there
  is unreachable rather than a decision — which is why `Indexing.axis_index_mentions_symbol` (the one
  shared "does this index depend on this symbol" predicate, with the set-of-symbols twin
  `axis_index_mentions_any`) counts a concatenation's iterators as mentioned: the constructor's own
  reading, safe to share downstream because nothing downstream can present one. Above lowering a
  `Concat` that DOES arrive is refused loudly, never approximated — `Indexing.reflect_projection` and
  `Assignments.apply_padding_offset` both raise, because neither a single affine stride nor a
  single-index shift can express a partition into disjoint sub-ranges (gh-ocannl-773).
- **Adding a `Low_level.t` constructor: let the compiler enumerate the exhaustive sites, and audit
  the catch-alls with warning 4** (gh-ocannl-696). Most walkers list every constructor, so the
  build fails at each (82 sites for `Scan_loop`); the ones it cannot flag are the `| _ ->` matches.
  `OCAMLPARAM="_,w=+4" dune build @arrayjit/lib/check` reports every fragile match with the type it
  is over — filter for `Low_level.t` (136 sites at the time) and read them in place: nearly all are
  shape recognizers whose fall-through is the right answer for a new statement kind, and the few
  descend-only walkers (`hoist_cross_statement_cse`, `proc_contains_set_from_vec`,
  `stmt_reads_cell`, the stored-computation scanners) need an explicit arm. Test-side walkers
  (`Ll_test.walk_t`, `bench_harness`, a few tests) surface only under the full `dune build @check`.
