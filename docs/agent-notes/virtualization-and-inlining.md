# Virtualization and inlining

Which candidates inline, where each rejection is decided, the policy caps, and how to test a
value-rewriting pass on hand-built IR.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Value-rewriting passes need executed parity tests, not just structural pins (see AGENTS.md).
  To exercise a virtualized affine-LHS producer end-to-end, hand-build an `Assignments.comp`
  (einsum result-side scatter specs don't parse; gradients accumulate → stay materialized): pass
  `~name` to `Context.compile` (or wrap in `Asgns.Block_comment` for labeled debug dumps), set
  `embedded_nodes`, force the output materialized, seed inputs with `Context.set_values`, then
  compile→run→`get_values`. When the shape under test is NOT reachable through `Assignments` at
  all (the optimizer never emits it), build the `Low_level.optimized` directly — `LL.optimize` over
  a hand-written `LL.t`, or `analyze_proc`+`specialize_proc` — and pass it as
  `Context.compile ?prelowered` with `~name` and `Ir.Assignments.empty_comp` (gh-ocannl-562).
  It replaces the compile's lowering wholesale, so the analysis layer and the kernels see one IR;
  add `~lowered_transform:(fun o -> [ o ])` to keep the default schedule annotator off hand-built
  code.
  See `test/operations/prelowered_seam.ml`, and mind the scope-purity contract below.
- Do not re-derive that harness: `test/support/ll_test.ml` (library `ll_test`, links `ocannl`) holds
  the LL builders, ONE exhaustive `Low_level.t`/`scalar_t` traversal with the counters derived from
  it, and `optimize`/`run`/`execute` — add `ll_test` to the test's `(libraries ...)`. A new IR
  constructor is handled there once instead of in every copy. `test_utils` stays separate on purpose:
  it depends on `arrayjit.ir` alone, for tests that link no more than that.
- Such a hand-built case gets its differential arm for free: pre-decide the placement in the
  `optimize_ctx` you hand `LL.optimize` (`Low_level.decide_materialized`, which is what
  `Context.decide_materialized` records for the `Assignments` pipeline; `Ll_test.optimize
  ~materialized` wraps it) and re-specialize the SAME `LL.t`. The inlined and materialized readings
  of one program must agree cell for cell, which is what pins a virtualization guard;
  `Context.decide_materialized` on the context itself cannot do this, because `?prelowered` replaces
  the lineage state with the record's own `optimize_ctx`.
  Three traps: (a) `known_non_virtual` does NOT mean "has a context buffer" — a node written and read
  within one routine and never observed is placed `Local`, routine-scoped scratch whose values never
  reach a context buffer. Host access to a node this lineage placed `Local` raises in BOTH
  directions (gh-ocannl-599; `test/operations/local_host_access.ml`), so the mistake fails loudly
  rather than reporting the uploaded copy back as a computed value. Read back only nodes you
  declared `On_device`, and mark them `Tn.set_observable` so the aliasing planner cannot hand their
  bytes to another node. (b) Producer/consumer indices that run past a node's dims are
  invisible while the node is virtual (the access is inlined away) and become real out-of-bounds
  traffic the moment the case executes or the materialized arm runs — size hand-built arrays for the
  materialized reading, and seed outputs with a sentinel so "wrote the wrong cells" fails the value
  check instead of reading whatever the buffer held. (c) The oracle has to discriminate, not merely
  exist: a producer must write a value that varies with EVERY symbol of its iteration and stays off
  the init value (`1 + i`, `1 + 10*outer + inner` — the `tick`/`tag` helpers in
  `test/operations/virtual_diagonal.ml`), because a constant producer just replays an identical
  assignment under a too-wide range guard, a value omitting a symbol is constant along that axis
  under a wrong substitution, and a value colliding with the zero-init hides a dropped first
  iteration.
- A `Local_scope` body's ONLY effect is on the locals it owns — its own scope id plus ids
  `Declare_local`d lexically within it (gh-ocannl-584). Not a tensor node, not a sibling's or
  enclosing scope's local, no `Workgroup_barrier`, no `Staged_compilation`. The reason is that a body
  does not execute where it is written, in three different ways: `C_syntax.pp_scalar` returns it as
  a local definition that `pp_local_defs` emits ahead of the enclosing statement ordered by
  `scope_id`; `simplify_llc` collapses a single-assignment scope into the expression, moving its
  reads the other way; and `hoist_cross_statement_cse` lifts a body shared by sibling statements to a
  top-level `Declare_local` + body, running ONCE ahead of the first user. Purity makes all three
  placements unobservable from outside the body, which is exactly what `Affine.path_before` assumes
  when it refuses to order sibling `Arg` positions; the two would otherwise disagree. Purity governs
  a body's EFFECTS, not its inputs — the hoist separately needs the body's reads untouched across
  the statements it is lifted over, and its hazard check must cover scope locals (`Get_local` /
  `Set_local`) as well as tensor nodes, or a shared body is lifted above a `Set_local` of a local it
  reads and later users read a stale value (a real miscompile, pinned by `prelowered_seam` phase 5).
  Bodies reading a local declared OUTSIDE them are ordinary pipeline output — CSE and the hoist
  itself create them — so "a body may only read locals it owns" is not available as a rule.
  `hoist_cross_statement_cse` is the ONLY pass that can move an effect out of a `Local_scope`
  (`simplify_llc`'s collapse cannot match an impure body; CSE's dedup leaves the surviving
  occurrence impure), so it guards its own precondition with `scope_purity_violation` and declines
  to hoist an impure body, instead of every public door into `specialize_proc` needing a gate. That
  is what keeps the raw analysis probes usable: an impure body reaching the pass would otherwise be
  laundered to a top-level `Declare_local` + body, and `Context.compile ?prelowered` would compile a
  silently changed routine. Rejected, never rewritten — pinned by `prelowered_seam` phase 6.
  The contract governs a body's EFFECTS only; it deliberately says nothing about the ORDER of its
  reads of other locals. A rule for that was written and reverted (gh-ocannl-584 review rounds 4-5):
  deciding "is this local emitted before me?" means replicating codegen's emission algorithm
  (`pp_local_defs` sorts by `scope_id`, per-statement def blocks, `Set_dynamic` concatenating two
  operands' defs), and every divergence is a FALSE REJECTION of valid IR — three surfaced in one
  review round, one of them rejecting CSE output inside a scope body, which the pipeline really
  produces. What the rule would have caught (a hand-built read of a sibling emitted later) fails
  loudly as a backend "use of undeclared identifier", not silently. If the guarantee is ever wanted,
  it belongs in `pp_local_defs`, which HAS the emission order in hand and so cannot diverge from
  it.
  `Low_level.validate_scope_bodies` enforces it at BOTH ends of the pipeline: `optimize_proc` on the
  way in (ahead of the analysis cache, so a digest hit cannot skip it — and before the hoist can
  launder a body write into a top-level statement that no later gate would recognize) and
  `C_syntax.compile_proc` on the way out (catching what a schedule transform constructs), the latter
  ahead of `validate_parallel_classified` and NOT transported as an `Illegal_schedule` — no schedule
  choice can rescue malformed IR. Its statement match is exhaustive with no catch-all, so a new
  `Low_level.t` constructor breaks the build until someone classifies it as body-legal or not. The
  pipeline complies by construction: `inline_computation` drops the inlined computation's `Set`s and
  `Zero_out`s. The raw analysis entry points `analyze_proc`/`specialize_proc` deliberately do NOT
  validate — they are the probes that must stay conservative on IR they may not trust
  (`test/operations/affine_extraction.ml`); everything past them does.
- The scope-TARGET contract, companion of the body contract above (gh-ocannl-681): a `Local_scope`
  over X denotes THE INLINED COMPUTATION OF X, so `LL.optimize` accepts one only while X is virtual,
  and REJECTS it over a materialized X (`scope_target_rejection` in `low_level.ml`, raised from
  `cleanup_virtual_llc`) instead of the silent normalization to a plain `Get` it used to do. The
  trap that makes this bite: a node with NO SETTER is decided non-virtual, so a hand-built scope
  over a freshly created node is a scope over a materialized node — declare it virtual
  (`Ll_test.virtualize`). Two shapes were green-by-collapse before the rejection: `accum_width.ml`'s
  gh-639 legs ran kernels literally spelling `acc[0] = acc[0]` (the identity copy reproduced the
  expected value), and `affine_extraction.ml`'s sibling-operand probe lost scope B's write while its
  own comment claimed the scopes survived `specialize_proc`.
  Exactly one exemption, and it is a retraction of the optimizer's OWN decision rather than of the
  caller's program: `virtual_llc` mints a scope at a `Get` of a still-virtual node, a later refusal
  can commit that node `Never_virtual`, and rewriting back to a `Get` is then sound because the
  surviving setter writes the value the body recomputed. `input_scope_ids`, taken before
  virtualization, is what tells the two apart — a scope in that set may not be rewritten away.
  The retraction is REACHABLE, and structurally so rather than by accident (gh-ocannl-704):
  `virtual_llc` walks statements in SOURCE ORDER while a node's placement is one mutable cell shared
  by the whole walk, so a refusal decided at a statement reached AFTER a read that already minted
  flips the node under an existing scope, and nothing revisits that scope in between. Both rejection
  families do it — store time (`check_and_store_virtual`, e.g. `Non_virtual 142` on a guarded LATER
  setter of an already-read node) and consumption time (`inline_computation`, `Non_virtual 13` at a
  second read the producer's index map cannot serve). Deleting the exemption therefore makes
  `LL.optimize` refuse IR it built itself; `test/operations/scope_over_materialized.ml` pins one
  witness of each family, each with executed parity against the same program's materialized reading.
  Both witnesses are hand-built, which is the exemption's honest standing: it is load-bearing for IR
  `optimize` accepts, not a mechanism user programs are known to hit — instrumented builds (the
  gh-ocannl-681 PR's, and a repeat over the targeted virtualization tests) recorded hits only on
  out-of-contract INPUT scopes.
  The SAME shape is legal and means the opposite AFTER `optimize`: `Schedule`'s materializing
  `Unroll` / `Partition` mints and `C_syntax.try_localize_serial_reduce` localize a materialized
  accumulator this way and codegen renders it. That asymmetry is the point — **materialized-accumulator
  localization belongs to codegen's accumulator peel (gh-ocannl-693) and to nothing else**; a second
  route through the virtualizer would restore the gh-639 "whichever schedule happened to run"
  problem. The peel is unconditional as of gh-ocannl-693 — every recognized serial reduction nest is
  localized, not only those whose storage precision the numerics policy wants widened — so ordinary
  lowering now produces this shape at f32 routinely; `test/operations/reduction_accumulator_residency.ml`
  pins it. Hand-built IR in that form still reaches a backend past the optimizer via
  `Ll_test.optimize_scoped` (optimize a scope-free raw twin for the traced store and placements,
  then swap in the scoped `llc`) and `Context.compile ?prelowered`. Pinned by
  `test/operations/scope_over_materialized.ml`.
  Since gh-ocannl-687 the node also RECORDS which side it came from — `Local_scope`'s `mint` field,
  `Inlined_computation` vs `Schedule_minted` — but that flag is deliberately not what decides this
  rejection, and claiming the schedule's provenance does not buy a program past the optimizer
  (pinned in the same test). The mint says which pass BUILT a scope, a durable fact consumers such
  as `Autotune.collect_loops` need; the rejection is about which side of a PARTICULAR `optimize`
  call a program was handed to, which only `input_scope_ids` can answer. Conflating them would let
  hand-built IR label its way back into the silent collapse. When building this shape by hand, spell
  the honest mint anyway — the canonical digest distinguishes the two, so an inlined scope wearing
  the schedule's label would key a different cache entry.
- A node-level "what happened at first touch" flag (`zero_initialized_by_code` and friends) cannot
  soundly drive a PER-OCCURRENCE codegen decision, because nothing clears it across the traversal: a
  guard keyed on it alone collapses `Zero_out; Set; Zero_out` to one zero and drops a `Zero_out`
  inside a `For_loop` on every iteration. The shape that works is per-traversal state — a `seen` set
  cleared at the single reset point (`compile_proc`) plus a positional `~in_loop` threaded through
  the recursion, defaulting to `true` for mutually-recursive callers that don't carry it. When a
  codegen decision consults a `traced_array`-style boolean, ask whether it is node-level or
  occurrence-level; they coincide only at first touch on the linear path.
- The codegen localizer forwards a preceding whole-node `Zero_out` directly into a serial
  accumulator local only when `Low_level.affine_accesses` finds exactly one same-cell RMW pair and
  `Affine.covers_box` proves its closing stores cover the whole node (gh-ocannl-821). The zero store
  is dropped only after `try_localize_serial_reduce` actually accepts, every loop that repeats a
  cell is inside that accepted scope, no enclosing loop is statically dead, and the covering write
  is unconditional. Those clauses are distinct from geometric coverage: an enclosing reduction
  loop can repeat an otherwise covering output nest, a dead loop executes no closing store, and a
  symbolic-extent guard can make only a prefix execute. A localizer/SIMD decline, partial or
  conditional coverage, an opaque effect, or an accumulation with no preceding zero retains the
  original opening value.
- **A virtualization candidate under an `If` is rejected, and the guard's position decides which
  arm rejects it** (gh-ocannl-651). `check_and_store_virtual`'s walk sees only the subtree it is
  handed, so a guard INTERIOR to that subtree hits its own `If` arm while a guard ENCLOSING it is
  invisible there — `virtual_llc` threads a `~guarded` flag down its walk and reports it, and both
  paths land on `Non_virtual 142`. Guards are NOT confined to the backend-compile-time launch-extent
  pass: `Assignments.to_low_level` emits interval guards for clamped-window pooling (gh-ocannl-504)
  and extent guards for symbolic extents (gh-ocannl-490), both before virtualization. When adding a
  pass that captures a subtree for later replay, ask what context the subtree was captured FROM —
  a walk of the subtree alone cannot see it.
- **The same question for loops: a candidate is captured at the outermost `For_loop` whose index
  occurs in its assignment indices** (`track_symbol` / `reverse_node_map`), so a reduction loop
  BELOW that point is part of the stored computation (the ordinary `x[t] += a[s]`, priced by
  `virtualize_max_inline_reduction`) while a repetition loop ABOVE it is not — and a symbol-free
  (all-`Fixed_idx`) index map has no capture site at all. Such a candidate is rejected as
  `Non_virtual 147` (gh-ocannl-674); width-1 loops stay exempt, since replaying one iteration once
  is exact. Two arms hide most of this shape and neither is a guarantee: an array reduction
  `x[0] += a[s]` is rejected because the sibling read escapes (`Non_virtual 9`), and an accumulator
  read more than `virtualize_max_visits` times is capped (`1:visit-cap`) — a flippable policy
  prior, decided in `decide_placements` before any legality question is asked.
- **Where a virtualization candidate is refused is readable off its placement PROVENANCE, and four
  phases write into the same table** (gh-ocannl-658, pinned row by row in
  `test/operations/virtual_rejection_boundary.ml`): `decide_placements` applies the heuristic caps
  (`Visit_cap` / uncovered read, `Inline_reduction_cap`, `Inline_fanin_cap`) BEFORE any legality
  question, so a shape capped there may be perfectly inlineable; `check_and_store_virtual` rejects
  at store time and `inline_computation` at consumption time — which codes each one mints is
  derived, not listed: `test/operations/nonvirtual_code_inventory.expected` groups them by function,
  and refuses a tag minted in two (the consumption-time backstop behind `148:scan-recurrence` is
  `149:scan-at-inline`, so the provenance says which phase spoke) — which is why two setters with
  different index maps as separate statements store fine as components and only fail once a read
  site cannot be served; and `cleanup_virtual_llc` commits a surviving read as `Surviving_read`,
  which is the absence of a rejection rather than one.
  Two properties of the store-time set are worth knowing before you chase one, and BOTH are read
  off pipeline order rather than off the arm. First, some of those arms cannot fire: nothing emits
  `Staged_compilation` (`Non_virtual 8`) today, and the passes minting barriers
  (`Non_virtual 141`), cooperative tiles (`Non_virtual 143`) and dynamic scatters
  (`Non_virtual 144` — `rewrite_one_hot_reductions`) all run after `virtual_llc`. Do not group arms
  by what they match on to predict this — `Scan_loop` (`Non_virtual 148`) and `If`
  (`Non_virtual 142`) are refused by constructor exactly as those are, and both fire on ordinary
  code.
  Second, **`Non_virtual 19` is live, and a note or comment telling you otherwise is stale.**
  `Assignments.lower` runs the algebraic rewrite tier — `Rewrites.apply`, gh-ocannl-483 — BEFORE
  `Low_level.optimize`, and its member `online_softmax` declares its cached probability cell as a
  `Declare_local` (`Online_softmax.hoist`, the consumer-side read hoist), so with that key on the
  arm fires for real; `row_hoisted_local` in the boundary test pins it on a plain hoisted local,
  since what the arm refuses is any `Declare_local` in the captured nest and not that rewrite's
  shape. Two qualifications: `online_softmax` is enabled by the `approximate` profile only —
  `performance` deliberately leaves the algebraic-rewrite gates alone as a numerics axis, and
  `reproducible` pins them off — and the rewrite's OTHER `Declare_local`, inside `emit_normalizer`'s
  scan body, never reaches 19, because an enclosing `Scan_loop` is refused as 148 first. (The
  running max/sum are `Scan_loop.carried` values, not locals.) Claims that
  `hoist_cross_statement_cse` is the only producer of `Declare_local` predate that tier.
  Both 142 and 148 arrive by two routes, and the pair is the general shape: the offending
  construct ENCLOSES the candidate's nest, so it is outside the captured subtree and only the
  caller can report it (`~guarded` / `~in_scan`, decided before the walk starts), or it sits
  INSIDE and the walk's own arm finds it. Same verdict either way — for 148 that one verdict also
  covers a scan the candidate merely reads from or sits beside (gh-ocannl-696).
  Pipeline order is what falsified 19's "cannot fire", and nothing tied the claim's copies
  together. So before moving a pass across `virtual_llc`, or adding one ahead of
  `Low_level.optimize`, read the golden of `dune build
  @test/operations/runtest-nonvirtual_code_inventory`: it lists every file that names a code — the
  raise-site comments, `docs/lowering_and_inlining.md`, this note, the tests — which is the
  checklist that change lacked (gh-ocannl-1015). Cite a code as `Non_virtual N` or by its tag, the
  two spellings the inventory reads. `Non_virtual 52` is enforced earlier still (`trace_node_facts`
  raises `invalid_arg` on a `Concat` write index, so the virtualizer's arm never sees one). The
  tags themselves, and how they compose, are the TAG entry below.
- **A placement provenance is a TAG, and which kind it is tells you whether code reads it**
  (gh-ocannl-609): `Tnode.provenance` splits two ways. A decision nothing interrogates is a
  `Site "<code>:<kebab-reason>"` explaining itself — some sixty of those, minted across nine modules
  from the C renderer's storage queries to the scheduler's tile placements. A tag some other code
  reads back is a `provenance` CONSTRUCTOR: `Visit_cap`, `Inline_reduction_cap`,
  `Inline_fanin_cap`, `Read_before_write`, `Scope_local`, `Surviving_read`. The split is a layering
  decision, not a style one — the type lives in `tnode.ml` (that is where `Placements` is), so a
  constructor per site would make the bottom module enumerate the vocabulary of every module above
  it, which is why the field was an unstructured `int` for years in the first place. The rule for a
  new tag: `Site` unless something matches on it.
  - The code is the integer the provenance used to be, so older issues and comments citing
    `Non_virtual 13` or "provenance 39" still resolve. Codes are NOT unique: `176`/`178` each name
    two different schedule sites, told apart by their reasons.
  - They compose STRUCTURALLY: `default_to_most_local` records
    `Refined (Inline_reduction_cap, Site "432:is-local-materialized-query")`, rendered
    `39:inline-reduction-cap -> 432:is-local-materialized-query` where the retired arithmetic wrote
    `39432`. `Tnode.leading_provenance` walks back to the first tag (what `Ll_test.rejection_code`
    returns); `provenance_to_string` is the only place the ` -> ` separator is spelled.
  - The two readers are `Low_level.is_cap_provenance` and `cap_provenance_setting`, both exhaustive
    matches with no wildcard: a fourth cap does not compile until its policy and its config key are
    stated. `cap_provenance_setting` is what lets the `Local` host-access refusal offer "raise the
    cap" instead of only "materialize the node" — the distinction the integer could not express —
    and the advice is HEDGED, because `decide_placements` records only the first cap that fires
    (each arm is guarded on the placement still being undecided) and the legality rejections run
    afterwards, so raising the named setting can merely expose the next obstacle.
- **A dynamic-gather table (`Get_dynamic`) materializes at the read, and a table declared `Virtual`
  is refused** (gh-ocannl-734, `test/operations/gather_table_placement.ml`): the gathered row is
  only known at runtime, so no computation can be replayed at the read site — `virtual_llc`'s
  `Get_dynamic` arm therefore commits an undecided table `Never_virtual 17` right there, exactly as
  the sibling lane-extract gather does for its packed-uniform counter (`Never_virtual 146`), and a
  table that is already `Virtual` gets a `User_error` naming the node, both readings and
  `set_materialized`. Both `Get_dynamic` arms (`virtual_llc`'s and `cleanup_virtual_llc`'s) carry
  that check, so neither can answer a hand-built gather with a bare provenance collision. Only
  hand-built IR reaches this: `Assignments` lowering emits no `Get_dynamic`, and the pipeline's own
  comes from `rewrite_one_hot_reductions`, downstream of both arms.
- **`virtual_llc` owns every placement decision for a tensor read it leaves behind**
  (gh-ocannl-805, `test/operations/virtual_decision_coverage.ml`). The always-on
  `validate_virtualization_decision_coverage` seam runs immediately after the pass and before
  cleanup: every `Get` / `Get_dynamic` in a statement cleanup will keep, including tensor-buffer
  reads nested in a kept inlined `Local_scope` body, must have an effective placement state (a
  lineage decision or declared intent). The intermediate IR still contains a virtual candidate's
  setter and its self-read, but cleanup drops that statement whole without visiting its RHS, so the
  seam mirrors cleanup's keep/drop decision and excludes it. `Never_virtual` counts -- it is the
  virtualizer's materialization verdict even though cleanup/backend finalization later resolves it
  to `Local` / `On_device`; a scope's `Get_local` and a `Get_merge_buffer` do not, because neither
  reads a tensor context buffer. This is deliberately a second exhaustive walk rather than an
  assertion copied into each read arm: forgetting a future arm fails at the pass boundary, before
  cleanup can make the result depend on whether it walks the undecided node's setter (`Virtual
  151/152`) or reader (`Never_virtual 17`) first.
- **A flip alternative's `fa_recompute_cost` is the cost model's count, not the traced proxy**
  (gh-ocannl-637 Part 2, gh-ocannl-1011): `Low_level.specialize_proc` prices each candidate through
  `Low_level.recompute_pricer`, a seam `Cost_model` registers at initialization (it sits above
  `Low_level`, so a hook is the only way the virtualizer can reach it — a program linking no cost
  model prices everything with the proxy). The price is `Cost_model.analyze` over CODE the inliner
  produced, never a rewrite of it: `Low_level.instantiate_at_synthetic_read` runs
  `inline_computation`'s own core (`instantiate_computations`) at a synthetic read — a fresh symbol
  per axis ranging over the axis, a fixed or static position mirrored — then the SAME
  `post_virtualization_pipeline` `specialize_proc` runs (cleanup through the cross-statement CSE),
  on a placement copy in which the node is virtual. `specialize_proc` hands the pricer a WORLD per
  node: a virtual node's stored computations under the placements as the walk left them (snapshot
  before cleanup — cleanup's commitments are the default reading's, and a flip re-makes them), or,
  for a cap-materialized node (never stored — the cap fires before the walk), `walked_computations`:
  the walk itself (`virtual_llc`) re-run over the node's RAW setter statements in a scratch copy of
  the lineage with the node undecided. The re-run decides everything the flip changes about the
  node's own computation — capture point, refusals, the raw storage of a packed-uniform producer, a
  footprint read its setter no longer hosts retracting — so imitating any of those rules is the
  wrong fix (an imitation of the capture rules missed both the raw storage and the retraction). No nested
  expansion: a stored template already carries the producers inlined when it was stored. Three
  traps: (a) pricing must not draw on `Indexing.get_symbol` or `get_scope` — generated code prints
  those counters, so a pricer that drew from them would renumber every later routine's goldens;
  `virtual_llc` and `inline_computation` take `?fresh_symbol`/`?fresh_scope`, and pricing passes the
  negative `pricing_symbol`/`pricing_scope`; (b) exactness is per leg (`rc_flops_approx` /
  `rc_bytes_approx`) and the seam gates on the op count only — a read under a guard's arm makes bytes
  a bound while ops stay exact (a diagonal producer's consistency guard), whereas a range guard's
  short-circuiting `&&` keeps the op count a bound (affine write positions, multi-setter
  components), exactly as for the emitted read; (c) a flip the store refuses (a scalar reduction's
  operand read escapes the setter it would be captured at, `9:`) is refused by the re-run walk and
  carries the proxy. The ordering is otherwise as before: `specialize_proc` multiplies the one-read
  count by the instantiations (read multiplicity, or `per_cell` for a footprint reading) and falls
  back to the proxy (reduction extent × multiplicity × transitive fan-in, `fa_modeled = false`) when
  the op count is only a bound. A test that stages a "decoy" or a budget cut on cost ORDER must build
  the order from modeled costs, not from fan-in counts; and a test that wants the visit cap needs a
  read off the reader's write position (a copy-position read is rmw-exempt, so two identity readers
  do not trip it). The relationship pinned is PRICE = EMITTED READ, per shape and through the seam,
  in `test/operations/cost_model_template.ml`.
- **The three caps have a cheaper landing spot than `Never_virtual`: footprint-scoped
  materialization** (gh-ocannl-616, `virtualize_footprint_materialization`, default on). When a cap
  would materialize a node whole but every read of it is an affine sub-image read — a diagonal
  `a[i, i]`, a slice, a cell read under loops its index does not mention — and the readers'
  iteration boxes together hold fewer cells than the node (`footprint_eligibility_query`, the
  profitability rule: both forms pay the same per-instantiation cost, so only the instantiation
  counts compare), `decide_placements` leaves the node undecided and records it in the
  specialization-local `footprint_scoped` table instead; `virtual_llc` then serves each read from a
  fresh scratch node in the `footprint` namespace (`Low_level.footprint_namespace`, how a test tells
  them apart) shaped like the READER's box, filled by a prologue that is the ordinary
  `inline_computation` instantiation at the read's indices under fresh loops. WHERE the prologue
  runs is the semantics: right after the top-level statement of a local producer's last write —
  the position its materialized buffer would have been complete at, so the scratch snapshots
  exactly what that buffer would have held even when a later statement rewrites a template input
  (review round 1 found the reader-side placement diverging from the cap's reading there) — and
  ahead of the reader's statement for an inherited template, the recompute-at-read reading
  inlining gives it, which is why an inherited template reading the reader's own target declines
  (`template_leaves`), why a reader placed before the producer's last write is ineligible, and why nothing but the producer may be written between its first and last write statements (a shared loop rewriting an input after the producer, a tensor OR A LOCAL rewritten between two
  accumulating components — local effects come from the routine's `Affine.statement_effect` rows,
  a local write inside the producer's own statement included, and a dead loop is no writer for it
  or for the prologue's position: the prologue replays every component after the last write;
  rounds 2-5). A read under a SCALAR gate (a `Where` arm, a gated operand) is not an `If` guard:
  `a_gated`, not `a_guarded`, marks it, and such a node is ineligible (round 7: the gate may be
  what keeps an instance in range, and the prologue is unconditional).
  A reader statement that writes a local is ineligible too (an inherited prologue, ahead of the
  statement, would see the local as it was; round 10). An explicit preference exempts its node from
  the caps even where the form cannot serve the reads; the node then inlines. Flip pricing is per
  read CELL in both readings (`per_cell`: the sites' fiber cardinalities, 1 per injective site),
  and an inherited footprint-scoped node offers its `` `Inline`` flip only. A
  consumption-time rejection after a scratch was minted is harmless: cleanup's scope-target
  retraction turns the stranded prologue into an n-cell gather of the buffer
  (`case_rejection_after_footprint`).
  Four more things that are easy to get wrong: (a) the decision is per ROUTINE, not a placement — a
  consumer routine footprint-scopes a node an earlier routine left `Virtual` on the template's own
  reduction extent (`template_facts`, over a SNAPSHOT of the traced store: reading the template
  registers operands the routine never mentions), the gh-573 corner, and the node stays
  `Virtual 152` in the lineage; (b) a read may be footprinted only in a MATERIALIZED
  consumer's setter, in a single-writer top-level statement, unguarded, outside a scan and outside
  a storage pass — anywhere else the read would land in a stored template (replayed by a later
  routine where the scratch does not exist), so the decision RETRACTS at that read: to the cap's
  own materialization for a local producer (recorded in `footprint_retracted`, so the decision
  surface stops offering the `` `Footprint`` flip), to plain inlining for an inherited node or an
  explicit `Context.decide_footprint` preference (exclusive with `decide_inline` per node —
  `prefer_inline` / `prefer_footprint` withdraw each other, so a search trying a node's sibling
  flips does not accumulate them); (c) the scratch's traced entry and its
  `Never_virtual 153` placement are minted in the virtualizer, ahead of `reconcile_traced_store`,
  which would otherwise read the scratch's write-then-read as a fresh node's read-before-write and
  demand it from a prior context; (d) `flip_candidates` is ONE record per node (gh-ocannl-1017):
  `fc_default` (the reading the policy chose) and `fc_alternatives` (the mutually exclusive
  readings a search may take — `` `Materialize``/`` `Inline`` for a footprint-scoped node, only
  `` `Inline`` for an inherited one, `` `Inline``/`` `Footprint`` for a cap-materialized node with
  a strictly smaller footprint), built in `specialize_proc` where both facts are known. The flat
  list it replaced cost nine review rounds across five consumers re-deriving the grouping, the last
  one an inference of the default from which records exist (an inherited footprint-scoped node and
  a cap-materialized one both offer a lone `` `Inline``). Consumers decide a candidate whole — keep
  the default or take one alternative — top-N cuts count candidates, and certainty bounds read
  `fc_default`, never the alternatives' shapes; `Autotune.placement_floor_withheld` is the one
  place a candidate's standing gates the floor.
  Structural probe for "inlined": `count_get` of the node, never `count_scopes` — the simplifier
  collapses a single-assignment scope into its expression. Pinned row by row, with executed parity
  against the materialized and (where the cap alone stands in the way) the inlined reading, by
  `test/operations/footprint_materialization.ml`; end to end through the default GPU schedule (the
  scratch crosses the prologue/reader statement boundary, so fission lifts it `On_device`, the
  two-launch form the issue ships first) by `test/operations/footprint_diagonal_einsum.ml`.
- Big-reduction producers are forced `Never_virtual` by `virtualize_max_inline_reduction`
  (default 16) — remember it when a structural expectation assumes inlining.
- Wide-fanin producers are forced `Never_virtual 41` by `virtualize_max_inline_fanin` (default 8,
  gh-573): a node whose fully-inlined computation would load more than that many distinct
  materialized nodes — accumulated through chains of virtual producers, per setter — materializes,
  resetting the fan-in downstream. This is the guard that breaks residual-stream-style running sums
  (per-cell multiplicity passes the visit cap because copy-position reads are rmw-exempt, yet each
  consumer re-sums the whole prefix); a structural expectation assuming a deep all-virtual chain
  must disable it (`Low_level.virtualize_settings.max_inline_fanin <- -1`). Like the other caps it
  is a flippable policy prior, not legality (`test/operations/virtual_chain_fanin.ml`). **The cap's
  bite depends on how many distinct transitive materialized inputs a chain accumulates — which varies
  with depth AND with graph shape at constant depth — so a cap conclusion holds for the graph it was
  measured on and not for a size class.** On gpt2_mini specifically (4 layers, gfx1151,
  `report-gh612-hip.md`), caps 16, 32 and −1 all emit a **135-kernel** arm A, yet only cap 32 is
  actually placement-identical: at cap 16 one node's worth of placement difference appears (the final
  layer norm gains a materialized `n792`) *behind an unchanged kernel count*. Node counts proxy guard
  firings; nothing logs provenance-41 decisions, so they are not firing counts. **Equal fission width can absorb a changed
  materialization decision, so a kernel count cannot establish that a cap did nothing — compare the
  emitted PARAMETER-SIGNATURE multisets and the materialized-node sets** (`benchmarks/gh612_cells.sh
  diff`, which needs only a snapshot). Three distinct levels, and conflating them is easy: a kernel's
  pointer parameters are exactly the materialized nodes it touches, so signature multisets track
  PLACEMENT and are insensitive to the crowned tile; kernel BODIES also move with the tile, so a body
  diff is not evidence of a placement change; and the count of newly materialized NODES is the proxy
  for guard firings, not the count of changed signatures — one materialization changes several
  consumers' parameter lists (on gpt2_mini, cap 8's 16/17 exclusive signatures come from 4 nodes:
  0/1/4/9/23 for caps 32/16/8/4/2). **A zero placement difference does NOT prove the guard was silent,
  so no fan-in bound follows from it**: `decide_placements` assigns provenance 41 only to a node not
  already placed, and `virtual_llc` afterwards rejects inlining for its own legality reasons, so with
  the cap disabled a different mechanism can materialize the same node and yield an identical source.
  The observable statement is all there is, and only at the caps actually swept (2, 4, 8, 16, 32, −1):
  placement differed from cap −1 at 2, 4, 8 and 16, and matched at 32. Nothing is established for
  caps between or above those. Cap 4 beat the default 8 by a
  non-overlapping 5.7% in a block order-balanced in BOTH the searches and the pass-2 replays (5.5%
  was the same six artifacts replayed in an unbalanced order; three replay sets of them spanned
  5.5-6.5%, so an identical schedule varies ~1pp run to run -- all three non-overlapping), and balancing that order matters: a fixed order
  confounds the cap with session position, which was worth ~1.4pp of an apparent 7.1%.
- **A candidate whose captured computation contains a `Scan_loop` is refused, and the refusal has
  two doors** (gh-ocannl-696, `test/operations/scan_loop.ml` legs 5 and 5b): `virtual_llc` threads
  `~in_scan` beside `~guarded`, so a per-statement store inside the scan hits `Non_virtual 148` in
  `check_and_store_virtual`, and a capture whose nest contains a scan anywhere — a sibling, or one
  feeding the value through a scope local — hits the same code from the validity walk's
  `Scan_loop` arm. The inline filter's own `Scan_loop` arm is a backstop raising 148, never a drop:
  dropping a value-producing scan returned the pre-scan value (Codex P1, staging#660 round 5). Reads inside the body inline as usual; the state
  node of a carried pair must be declared virtual (`Ll_test.virtualize`; the validator names it
  otherwise) and cleanup commits it `Virtual 16` like a scope local's node. The contract itself —
  one node DECLARED virtual per pair, never accessed as a tensor buffer, ids pairwise distinct,
  rebound by no `Declare_local`/`Local_scope` inside the scan and referenced nowhere outside it, inits free of carried state and of
  the scan index, `next` written exactly once at the body's top level and read only by later
  statements (it is declared without a value), no write of `prev`, no `Staged_compilation` inside,
  and a NON-EMPTY range (a dead scan is refused rather than given a meaning: review rounds 4–8 on
  staging#660 found a fresh walker each round that needed its own "dead scan is a no-op"
  convention, so the class was closed by making the shape unreachable) — is
  `Low_level.validate_scan_loops`, run at both gates like scope purity. Codegen's three per-local
  censuses (rng precision, accumulator residency, controlled accumulators) walk the implicit
  `prev = init` / `prev = next` assignments as the `Set_local`s they render as
  (`C_syntax.scan_implicit_set_locals`), and the carried ids are pinned to their node's STORAGE
  precision in `scope_prec_of` (`carried_state_scope_ids`, rng-carve-out precedence): a half state
  rounds every step — `scan_loop.ml` leg 6b holds 2048 through six +1 steps where a single state
  reaches 2054. Build scans through `Ll_test.carry`/`scan`/`prev`/`next`/`set_next`.
