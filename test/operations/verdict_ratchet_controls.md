# `verdict_ratchet` synthetic controls: the mutation-run manifest

`test/operations/verdict_ratchet.ml` carries its own negative and near-miss controls (the
`quantified_helper_controls` list and the `run_*_control` families). A control earns its place by
a **mutation run**: the scanner mechanism it pins is deliberately disabled, the focused alias
`dune build @test/operations/runtest-verdict_ratchet` is run, and exactly the intended control(s)
fail while their accepting neighbours stay green. Those runs were made through `tools/test-run.sh`
while landing staging PR #633 (gh-ocannl-887, gh-ocannl-891), and their evidence was cited only in
the PR's inline review replies. This file is the one place that collects them, so a scanner change
can find every control that guards the mechanism it touches and re-run the same mutation.

Conventions:

- **Round** is the *Review fixes round N* commit on `master` that introduced the mechanism and its
  control, by the id the commit has on `master` (PR #633 was rebased before it merged, so the ids
  its review thread cites for rounds 1-17 resolve nowhere in the merged history; `git log
  --grep='Review fixes round N:' -- test/operations/verdict_ratchet.ml` is how to look one up).
- **Mutation** is what was disabled or reverted in the scanner for the run; the control names are
  the exact labels in `quantified_helper_controls`, which is also how they print in
  `verdict_ratchet.expected` under "Synthetic helper-rule controls:".
- **Run** is the `tools/test-run.sh` run id. The log lives at `~/.ocannl-test-runs/<run>/log` on
  the machine that ran the review (`lukstafi`'s macOS box, worktree `ocannl-887`); the id is the
  durable citation, the path is machine-local. Every mutation run exited 1 with exactly the listed
  controls reporting `false` (verified 2026-09-05 against the retained logs); the confirmation
  runs exited 0.
- A control listed here is **maintained**, and the ratchet checks it on every run: a backticked
  phrase containing a space in this file names a control (the one exception is a phrase starting
  with `dune`, a command), every such phrase must be a control label the ratchet prints under
  "Synthetic helper-rule controls:" in its golden, every label printed there, the
  `run_*_control` families included, must appear in this file, and each exactly once: a control
  belongs to one row, and a second row citing it names its mechanism only in prose. Renaming a control on one side only fails the run
  with the offending names on stderr. When a scanner change makes a mutation no longer meaningful,
  replace the row rather than deleting it.

## Mutation runs with a retained log

| Round | Mutation applied to the scanner | Control(s) that failed under the mutation | Run |
|---|---|---|---|
| 5 `b9d4f58ac` | Match-bound return values not mapped back to the scrutinee producer | `refuses a helper that returns a match-bound quantified value` | `20260904T165054Z-79197` |
| 5 `b9d4f58ac` | Recursive binding group analysed as non-recursive (no sibling fixed point) | `refuses a quantified helper reached through a mutually recursive sibling` | `20260904T165128Z-90221` |
| 5 `b9d4f58ac` | Function parameters left in the outer helper lookup while the body is analysed | `does not resolve an outer quantified binding shadowed by a function parameter` | `20260904T165229Z-31857` |
| 6 `5eb4dceff` | `Fn.id`/`Fun.id`/`Stdlib.Fun.id` not treated as transparent Boolean wrappers | `refuses a fully applied quantifier through a transparent Boolean wrapper` | `20260904T171029Z-46614` |
| 6 `5eb4dceff` | `returned_binding_polarities` resolves names past an intervening `let` | `does not return a quantified binding shadowed by a later local` | `20260904T171113Z-53808` |
| 6 `5eb4dceff` | `filter`/`filter_map` views collapsed to the source name as population identity | `refuses a guard on a differently filtered population` (the same-filter control stays accepted) | `20260904T171139Z-60506` |
| 7 `d5c432535` | Dependency collection skips `if` conditions and protected `try` bodies | `refuses a bound quantifier returned through an if condition`; `refuses a bound quantifier returned from a protected try body`; also flags the `shell_scripts_parse.ml:line_enables_errexit` exemption as no longer earned, proving that exemption live | `20260904T172446Z-69949` |
| 8 `59a228a52` | Match/try case patterns not removed from the outer helper environment | `does not resolve an outer binding shadowed by a match pattern` | `20260904T173912Z-15580` |
| 8 `59a228a52` | Optional-default dependency edges disabled | `resolves an outer quantified binding used by an optional default` | `20260904T173938Z-31569` |
| 8 `59a228a52` | Match-case guard traversal disabled | `refuses a bound quantifier returned through a match guard` | `20260904T174007Z-50752` |
| 9 `df1232140` | Supplied-optional-label set cleared at helper calls | `does not use an optional default dependency when the caller supplies the argument`; `suppresses an earlier default inside a later default when the caller supplies it` | `20260904T175423Z-47445` |
| 9 `df1232140` | Earlier optional defaults removed from the environment of later defaults | `resolves a quantified binding through chained optional defaults` | `20260904T175458Z-67152` |
| 9 `df1232140` | Direct condition quantifiers not attributed through complementary Boolean `if` branches | `refuses a direct quantifier returned through an if condition` | `20260904T175530Z-87146` |
| 10 `c34125adf` | Every `?label` forwarding treated as supplying the optional | `uses an optional default when a forwarded argument is None` (forwarded `Some` stays accepted) | `20260904T180755Z-76190` |
| 10 `c34125adf` | Direct quantifiers in match guards not collected | `refuses a direct quantifier returned through a match guard` | `20260904T180830Z-85990` |
| 10 `c34125adf` | `Pfunction_cases` bodies not traversed for returned quantifiers | `refuses a quantified helper written with function-case syntax` (its guarded neighbour stays accepted) | `20260904T180904Z-95866` |
| 11 `2c2add4ed` | Non-empty witnesses unioned across `function` cases | `does not share a function-case guard with another case` | `20260904T182203Z-82716` |
| 11 `2c2add4ed` | Match/try patterns not shadowed in `returned_binding_polarities` | `does not return an outer quantified local shadowed by a match pattern` | `20260904T182252Z-93990` |
| 12 `4a92d6d4f` | Scrutinee attribution through complementary `true`/`false` constructor matches disabled | `refuses a direct quantifier forwarded by a Boolean constructor match` | `20260904T183621Z-80904` |
| 13 `816675a71` | Local Verdict claim wrappers not recognised | `refuses a quantified binding passed through a Verdict wrapper`; `refuses a direct quantifier passed through a Verdict wrapper` | `20260904T185240Z-12119` |
| 14 `9c25abea2` | Wrapper slot signature replaced by an empty one | the two round-13 wrapper controls plus `refuses a direct exists negated by a labeled Verdict wrapper parameter`; `refuses a bound exists negated by a labeled Verdict wrapper parameter` | `20260904T190912Z-77325` |
| 14 `9c25abea2` | `try` handler guards not analysed for direct quantifiers | `refuses a direct quantifier returned through a try-case guard` | `20260904T190946Z-84558` |
| 15 `7d98daa1f` | `boolean_match_polarity` not composed with the claim polarity for bound scrutinees | `refuses a bound exists inverted by a Boolean constructor match` | `20260904T192620Z-99615` |
| 15 `7d98daa1f` | `try` handler guard dependencies not signed by the handler result | `refuses a bound exists inverted through a try-case guard` | `20260904T192722Z-34041` |
| 15 `7d98daa1f` | Wrapper optional-default traversal disabled | `uses an omitted optional default that feeds a Verdict wrapper claim` | `20260904T192814Z-66163` |
| 15 `7d98daa1f` | Tuple/record bindings do not retain unguarded quantified components | `refuses a quantified component destructured from an intermediate aggregate` | `20260904T192852Z-77163` |
| 16 `65292bc31` | Unknown forwarded `?opt:expr` treated as `None` | `inspects the possible payload of an unknown forwarded wrapper option` | `20260904T195049Z-54546` |
| 16 `65292bc31` | `Verdict.p` applied to its label alone records no owed Boolean slot | `refuses a direct quantifier passed to a partially applied Verdict claim` | `20260904T195124Z-76928` |
| 16 `65292bc31` | Nested-`let` witness boundary not sealed against outer same-spelled witnesses | `keeps an outer witness from guarding a shadowed nested population` | `20260904T195208Z-90599` |
| 17 `bd67f0778` | Owed Boolean slot created only for parameterless partial wrappers | `refuses a direct quantifier passed to a curried partial Verdict wrapper` | `20260904T200334Z-43193` |
| 17 `bd67f0778` | Case-guard polarity falls back to positive-only when the result is not a literal | `refuses a direct match guard whose false result is a Boolean alias`; `refuses a bound match guard whose false result is a Boolean alias` | `20260904T200408Z-50550` |
| 24 `d230df573` | Match-case returned quantifiers not sealed before re-entering an outer Boolean | `does not let an outer guard witness a match-bound population` | `20260904T215904Z-75791` |
| 24 `d230df573` | Wrapper-return polarity extraction ignores the `if` condition | `refuses a quantified condition used as a wrapper claim value` | `20260904T215941Z-83437` |
| 24 `d230df573` | Qualified lookup of file-local module wrappers broken | `refuses a quantified argument passed to a qualified local-module wrapper` | `20260904T220149Z-91421` |
| 25 `8c090d393` | Opened-module prefix corrupted on import of local wrapper exports | `refuses a quantified argument passed through an opened local module` | `20260904T221532Z-96093` |
| 25 `8c090d393` | Returned match-pattern bindings not mapped to the scrutinee for wrapper slots | `refuses a quantified argument forwarded through a match wrapper` | `20260904T221610Z-4324` |
| 25 `8c090d393` | Wildcard cases ignored when deriving Boolean match polarity | `refuses a direct quantifier forwarded by a wildcard Boolean match` | `20260904T221501Z-88511` |
| 26 `5b5cba542` | Local modules export only claim wrappers, not quantified helpers | `refuses a quantified helper called through a local module`; `refuses a quantified helper called through an opened local module` | `20260904T222946Z-71923` |
| 26 `5b5cba542` | Immediately invoked function body not traversed on the direct-quantifier path | `refuses a direct quantifier returned by an immediately invoked function` | `20260904T223017Z-79747` |
| 26 `5b5cba542` | Immediately invoked function body not traversed on the named-dependency path | `refuses a quantified binding returned by an immediately invoked function` | `20260904T223101Z-87728` |
| 26 `5b5cba542` | Quantifier-function alias lookup broken | `refuses a direct quantifier called through a function alias` (the inverted near-miss stays accepted) | `20260904T223205Z-12199` |
| 27 `1e14b1368` | Boolean-match result not applied to the scrutinee before deriving wrapper slots | `refuses a quantified argument forwarded by a Boolean match wrapper` | `20260904T224356Z-80717` |
| 27 `1e14b1368` | Callback bodies not traversed during wrapper claim discovery | `refuses a quantified argument claimed inside a callback` | `20260904T224435Z-88688` |
| 27 `1e14b1368` | Callback parameter shadowing done by filtering instead of lexical tombstones | `does not connect a callback-shadowed parameter to its wrapper` | `20260904T224511Z-96391` |

### The deferred prototype

| Round | Prototype | What it showed | Run |
|---|---|---|---|
| 25 `8c090d393` | Native `Verdict.*` calls scanned for direct quantifiers (not merged) | Caught the proposed fixture, and also 65 clean-tree corpus sites plus three claim-local controls now reporting `Verdict.p` as the helper. Deferred to ahrefs/ocannl#908 as an audited migration, which landed with the gh-ocannl-931 provenance layer: the corpus sites were migrated to the `p_*` combinators or exempted by claim label, one decision per site, and a direct quantifier reaching a native claim is named by the claim's label. | `20260904T221248Z-50429` |

## The provenance layer (gh-ocannl-931) and its precision controls (gh-ocannl-908)

`test/support/verdict_provenance.ml` replaced the ratchet's separate walkers with one
scope-and-polarity model, so most of the mechanisms above are now one rule each of that model
rather than a path of their own. The controls this round added pin the model's new behaviour; the
mutation for each was made in the provenance module and run through the focused alias like the
rounds above (runs of 2026-09-10, on the same box; the corpus was clean before each run, and each
run's log ends with exactly the controls the row names reporting `false`).

| Mechanism | Mutation applied to the provenance layer | Control(s) that failed under the mutation | Run |
|---|---|---|---|
| Witnesses swap polarity with the value, so a guarded conjunction negated into a binding keeps its population witnessed when read back through `not` (gh-ocannl-908 item 1); coverage is decided where witness and source meet, so a negation has nothing to undo | `negate` drops the swapped views' witnesses -- which is the single-negation guard of a negated is_empty too, so this mutation fails every guarded control, `accepts a guarded quantifier assigned through a double negation` among them, and 105 corpus sites (`refuses a double negation without the non-empty witness` and `refuses a double negation guarded over a different population` stay refused) | `20260910T101641Z-78070` |
| A conditional whose alternatives all return one Boolean is that constant, through aliases and nesting (item 2) | Constant agreement across alternatives not folded | `does not attribute a condition whose branches return the same Boolean alias`; `does not attribute a condition whose branches agree through a nested condition`; the round-18 same-literal control; and the four matrix rows whose shadowed case is a constant conditional (`refuses a condition with one constant branch and one the reader cannot prove` stays refused) | `20260910T101648Z-83242` |
| A condition steers between values and is the value only where a branch returns a constant | (design rule, exercised on every run) | `does not attribute a condition steering between two unproven branches`; `accepts a witness from a condition selecting the claimed branch` | -- |
| Function parameters are bindings in the one environment, so a wrapper body's claim scanning sees the parameter and not the outer binding it shadows, while an optional default is walked before its parameter is bound (item 3) | (structural in the model: there is no second environment to leave unshadowed; exercised on every run) | `accepts a constant argument through a parameter shadowing an outer quantified binding`; `still refuses the outer quantified binding a wrapper closes over`; `evaluates a wrapper's optional default before its parameter shadows the name` | -- |
| A destructured parameter receives the component of the actual argument its name was bound to (item 4) | `projected_arguments` hands every name the whole actual | `accepts a quantified sibling ignored by a destructured wrapper parameter`; `accepts a quantified field ignored by a record wrapper parameter`; the destructured-parameter matrix row (`refuses the claimed component of a destructured wrapper parameter`, `refuses the claimed field of a record wrapper parameter` and `conservatively inspects a wrapper argument its pattern cannot align` stay refused) | `20260910T101655Z-88455` |
| Native claims are closures like local wrappers, so a quantifier written directly in `Verdict.p`'s argument is found and named by the label (item 5) | Claims fired by a native closure not emitted | `refuses a quantifier written directly in a native claim`; `refuses a quantifier written directly in a computed-label native claim`; `refuses a quantifier piped into a native claim`; `refuses a quantifier written directly in an opened native claim`; `names a native claim's non-literal label by its expression`; `the shipping ratchet process refuses a quantifier written directly in a native claim`; every matrix row (each fires through a native claim), the wrapper-call shadowed-quantified controls, and the label-keyed exemptions reported stale (`accepts a guarded quantifier written directly in a native claim` and `accepts a negated quantifier written directly in a native claim` stay accepted) | `20260910T101701Z-93611` |
| An open of List or Array, a module alias of either, and `List.(…)` bring the quantifier functions into scope by name (item 6) | `quantifier_exports` exports nothing | `refuses a quantifier reached through an open of List`; `refuses a negated exists reached through an open of Array`; `refuses a quantifier reached through a module alias`; `refuses a quantifier reached through a local open`; `refuses a quantifier reached through a let-open of Array`; and every matrix row, through its opened, aliased and local-open columns (`accepts an inverted quantifier reached through an open of List`, `does not resolve an unqualified for_all with no open in scope` and `keeps a local open of List inside its scope` stay accepted) | `20260910T101708Z-98774` |
| A helper's population parameters are substituted by the actual arguments, so a caller's witness covers the helper's quantifier over the same actual and a helper's own witness covers the caller's quantifier | Population substitution at application disabled | `accepts an outer guard forwarded to a helper call over the same actual`; `accepts a witness a helper establishes over the same actual` (`refuses a helper applied to an expression that is not a population` and `refuses a witness a helper establishes over a different actual` stay refused) | `20260910T101714Z-4598` |

## Mutation runs recorded only in the PR thread

These rounds report the mutation in their inline review reply on staging PR #633 but retain no run
id. The control names are the ones the round added; re-running the mutation is the way to
re-establish them.

| Round | Mutation reported | Control(s) |
|---|---|---|
| 1 `ae1b06a51` | Explicit true/false comparison polarity disabled | `refuses a fully applied quantifier compared with true`; `accepts a fully applied quantifier compared with false` |
| 1 `ae1b06a51` | Lexical environment for `let` bindings nested inside a claim argument removed | `refuses a binding nested directly inside a claim argument` (guarded and negated neighbours stay accepted) |
| 2 `ec34361be` | Direct `Bool.equal` dispatch removed | `refuses a direct Bool.equal true around a fully applied quantifier`; `accepts a direct Bool.equal false around a fully applied quantifier` |
| 2 `ec34361be` | Intermediate dependencies collected without polarity; inherited guards not propagated | `accepts a negated intermediate binding`; `accepts a guarded intermediate binding` |
| 2 `ec34361be` | Literal tuple/record pattern mapping to producers disabled | `refuses a quantified component of a destructured tuple binding`; `conservatively refuses a quantified component of a record binding` |
| 3 `dca4325db` | Inherited guards forwarded across helper calls (superseded: the provenance layer of gh-ocannl-931 substitutes a helper's population parameters by the actual arguments, so the same-actual guard is now accepted, by the control the gh-ocannl-931 table lists under population substitution, while the mismatched actual stays refused; the mutation for the substitution is in the gh-ocannl-931 table below) | `refuses a mismatched actual hidden by equal formal names` |
| 4 `dcf717100` | Inherited guard identity forwarded into a nested `let` | `refuses a shadowed guard identity across a nested alias` |
| 4 `dcf717100` | Pipeline-position `not`/`Bool.not` not recognised | `accepts a piped negated intermediate binding`; `accepts a directly quantified value piped through not` |
| 4 `dcf717100` | Negative dependency edges dropped | `refuses a negated bound exists` (`accepts a positive bound exists` stays accepted) |
| 4 `dcf717100` | Signed local returns removed, in each direction | `accepts a helper that negates a quantified local binding`; `refuses a double negation around a quantified local binding` |
| 18 `eb72b88a0` | Constant Boolean alias resolution for `if` polarity disabled | `refuses a direct if condition whose false outcome is a Boolean alias`; `refuses a bound if condition whose false outcome is a Boolean alias` (`does not attribute a condition whose branches return the same literal` stays accepted) |
| 18 `eb72b88a0` | Tail-position setup unwrapping (let/sequence/local open/constraint/coercion) disabled | `refuses a direct quantifier passed through a wrapper with tail setup`; `refuses a returned quantifier behind a local open` |
| 18 `eb72b88a0` | Polarity flip through `not` applied with `@@` removed | `refuses a negated exists written with the application operator` |
| 18 `eb72b88a0` | Direct wrapper quantifiers keyed by the wrapper definition offset | the "reused wrapper call" shadowed-quantified controls (`run_shadowed_quantified_controls`) |
| 19 `5a04464da` | Scoped alias map dropped from wrapper tail traversal | `refuses a direct quantifier passed through a wrapper setup alias` (`does not connect a wrapper parameter hidden by a setup constant` stays accepted) |
| 20 `539804bc7` | Constructor-match polarity without constant-alias resolution | `refuses a bound exists inverted by aliased Boolean match outcomes`; `refuses a direct exists inverted by aliased Boolean match outcomes` |
| 20 `539804bc7` | Boolean comparison without constant-alias resolution | `refuses a direct exists compared with a false Boolean alias`; `refuses a bound exists compared with a false Boolean alias` |
| 21 `16839405f` | Unsupplied claim slots not forwarded through a partially applied local wrapper | `refuses a direct quantifier passed to a partially applied local wrapper` |
| 22 `b9cdffab2` | Direct wrapper quantifiers keyed by the call offset instead of the argument offset | the "call slots" shadowed-quantified controls (`run_shadowed_quantified_controls`) |
| 23 `5e92676b0` | Local-module/local-exception setup not recursed into for returned quantifiers | `refuses a returned quantifier behind local module setup` |

## Confirmation runs

Green runs cited beside the mutations, on the rebased head of the round: the focused ratchet alias,
then `@test/operations/scans`, then `@check`.

| Round | Ratchet (and co-migrated aliases) | Scans | `@check` |
|---|---|---|---|
| 6 | `20260904T171421Z-23314` (with `launch_predicate_parity`, `config_usage_scan`, `dead_export_scan_cases`, `env_var_deps`) | | |
| 13 | `20260904T185518Z-88373` (with `test_random_histograms`, `threefry4x32_demo`) | `20260904T185630Z-26420` | `20260904T185651Z-35075` |
| 14 | `20260904T191034Z-91792` | `20260904T191059Z-98768` | `20260904T191118Z-6656` |
| 15 | `20260904T193638Z-84803` (with `autotune_arm_containment`, `test_random_histograms`, `threefry4x32_demo`) | `20260904T193649Z-91822` | `20260904T193704Z-8077` |
| 16 | `20260904T195243Z-886` | `20260904T195304Z-9381` | `20260904T195321Z-18572` |
| 17 | `20260904T200448Z-57923` | `20260904T200503Z-65115` | `20260904T200518Z-72328` |

## The other controls, exercised on every run

Every remaining label in `quantified_helper_controls`, grouped by the commit that added it: the
accepting neighbours of the mutation rows above (the near-miss that must stay accepted while its
refusal twin fails), the refusals whose review reply reported the fix without a separate mutation,
and the controls that predate PR #633. None has a one-off mutation run behind it; each is
exercised by every `dune build @test/operations/runtest-verdict_ratchet`. A round number is a
*Review fixes round N* commit of PR #633; a bare id is an earlier commit.

| Introduced by | Controls |
|---|---|
| `b0f17a019` (Ratchet helper-wrapped quantified claims) | `refuses an unguarded for_all2_exn helper behind a local Verdict alias`; `refuses a sibling for_all helper through an intermediate result binding`; `refuses an is_empty helper whose claim can pass on an empty source`; `refuses a negated exists helper with the same empty-population hole`; `accepts the explicit non-empty guard installed by the parity sweep`; `does not let a guard on somebody else's population answer for the helper`; `accepts a positive literal length as the non-empty witness`; `accepts a negated for_all2_exn discrimination helper`; `accepts a positive exists helper, which is false on an empty population`; `ignores a quantified helper that reaches no Verdict claim` |
| `4598a029b` (Close quantified helper analysis gaps) | `refuses a helper that returns a fully applied quantified local binding`; `refuses a quantifier written in pipeline style`; `keeps helper resolution inside its lexical scope`; `refuses a reversed length upper bound masquerading as a witness`; `preserves positive polarity through comparison with false` |
| `b9775c3ce` (Add the open-oriented Verdict.Claims surface (gh-ocannl-815)) | `refuses an unguarded helper behind an open of Verdict.Claims`; `keeps an open of Verdict.Claims inside its local scope` |
| `7c4e8a826` (verdict_ratchet: close bound-value and site identity holes) | `refuses a fully applied quantifier bound before the claim`; `accepts a fully applied quantified binding with a non-empty witness`; `accepts a negated fully applied quantified binding` |
| 1 `ae1b06a51` (preserve claim polarity and nested scope) | `accepts a guarded binding nested directly inside a claim argument`; `accepts a negated binding nested directly inside a claim argument` |
| 2 `ec34361be` (retain binding semantics through indirection) | `refuses a positive intermediate binding` |
| 5 `b9d4f58ac` (close remaining binding scope gaps) | `still resolves a non-shadowed quantified binding returned by a function` |
| 6 `5eb4dceff` (preserve returned population identity) | `accepts a quantifier guarded by the same filtered population` |
| 7 `d5c432535` (trace returned control-flow dependencies) | `accepts an inverted bound quantifier returned through an if condition` |
| 8 `59a228a52` (scope match and optional bindings) | `accepts an inverted bound quantifier returned through a match guard`; `preserves polarity through an optional default` |
| 9 `df1232140` (make optional defaults call-aware) | `accepts an inverted direct quantifier returned through an if condition` |
| 10 `c34125adf` (cover forwarding and function cases) | `accepts an inverted direct quantifier returned through a match guard`; `does not use an optional default when a forwarded argument is definitely Some`; `accepts a guarded quantified helper written with function-case syntax` |
| 12 `4a92d6d4f` (trace Boolean constructor matches) | `accepts a direct quantifier inverted by a Boolean constructor match` |
| 13 `816675a71` (trace Verdict wrapper arguments) | `accepts a guarded direct quantifier passed through a Verdict wrapper`; `accepts a negated direct quantifier passed through a Verdict wrapper` |
| 14 `9c25abea2` (map claim wrapper formals) | `accepts a positive exists passed through a labeled Verdict wrapper parameter`; `accepts an inverted direct quantifier returned through a try-case guard` |
| 15 `7d98daa1f` (retain remaining claim dependencies) | `does not use a Verdict wrapper default when its argument is supplied` |
| 16 `65292bc31` (close forwarded and scoped claim gaps) | `accepts a nested quantified population with its own witness` |
| 19 `5a04464da` (retain local and wrapper aliases) | `refuses a direct if condition whose local false outcome is a Boolean alias` |
| 20 `539804bc7` (complete wrapper and Boolean alias coverage) | `refuses a direct quantifier passed to a formatted partial Verdict wrapper`; `refuses every quantified argument passed through a sequential wrapper` |
| 21 `16839405f` (follow wrapper branches and partials) | `refuses a quantified argument claimed inside wrapper control flow`; `refuses an aliased quantified condition in a protected try body` |
| 22 `b9cdffab2` (expose nested wrapper claims) | `refuses a quantified argument claimed inside an eager wrapper call`; `refuses a quantified argument claimed under a local Verdict open` |
| 23 `5e92676b0` (trace function cases and local setup) | `uses a Verdict wrapper default preserved through partial optional None`; `refuses a quantified argument claimed by a function-case wrapper` |
| 24 `d230df573` (isolate scopes and export local wrappers) | `accepts an inverted quantified condition used as a wrapper claim value` |
| 25 `8c090d393` (trace opened and matched wrappers) | `accepts a direct quantifier inverted by a wildcard Boolean match` |
| 26 `5b5cba542` (retain callable quantifier origins) | `accepts a negated quantifier returned by an immediately invoked function`; `accepts a negated quantifier called through a function alias` |
| 27 `1e14b1368` (trace match and callback claims) | `accepts a quantified argument inverted by a Boolean match wrapper` |

## The `run_*_control` families

These predate PR #633 and are exercised on every run rather than by a one-off mutation; every
label they print is listed so the inventory the ratchet checks is complete.

- `run_refusal_controls` (`--quantified-helper-refusal-control`,
  `--direct-quantifier-refusal-control`): the ratchet re-executes itself on a synthetic offending
  fixture and checks the exact refusal diagnostic and exit status -- one child per diagnostic, the
  bound-helper one introduced by `cf1874075` (bind refusal markers to exercised controls) and the
  direct-quantifier one with gh-ocannl-908. Labels:
  `the shipping ratchet process refuses the planted helper fixture` (listed here; its twin is in
  the gh-ocannl-908 table above).
- `run_stale_quantified_control`: an exemption whose key no longer matches a live claim is refused.
  Introduced by `ad983d863`. Label: `refuses a stale quantified-helper exemption`.
- `run_shadowed_quantified_controls`: an exempted quantified helper key must name exactly one
  definition, including two definitions on one physical line, one call site, or one call slot.
  Introduced by `ad983d863`; the wrapper call-site and call-slot members by rounds 18 and 22 above.
  Labels, a resolution claim and a refusal claim per fixture:
  `a shadowed helper name resolves to two definitions, not one`;
  `refuses an exemption key that names both shadowed definitions`;
  `a same-line shadowed helper name resolves to two definitions, not one`;
  `refuses an exemption key that names both same-line shadowed definitions`;
  `a reused wrapper call helper name resolves to two call sites, not one`;
  `refuses an exemption key that names both reused wrapper call call sites`;
  `a multi-slot wrapper call helper name resolves to two call slots, not one`;
  `refuses an exemption key that names both multi-slot wrapper call call slots`.
- `run_colliding_site_controls`: literal-label and computed-label exemption keys that resolve to
  two sites, on separate lines or twice on one line, are refused. Introduced by `7c4e8a826`
  (gh-ocannl-891 offset-based identities). Labels, the same pair per fixture:
  `a repeated literal-label exemption key resolves to two source sites, not one`;
  `refuses an exemption key that names both repeated literal-label source sites`;
  `a same-line repeated literal-label exemption key resolves to two source sites, not one`;
  `refuses an exemption key that names both same-line repeated literal-label source sites`;
  `a repeated computed-label exemption key resolves to two source sites, not one`;
  `refuses an exemption key that names both repeated computed-label source sites`;
  `a same-line repeated computed-label exemption key resolves to two source sites, not one`;
  `refuses an exemption key that names both same-line repeated computed-label source sites`.
- `run_manifest_controls`: this file against every label above, both directions, as the
  conventions state, after holding the labels themselves duplicate-free (two controls under one
  label would be one row here and one golden line, so the second identity would vanish before
  either direction is checked). Labels: `every synthetic control has a row in the mutation-run manifest`;
  `every control phrase in the mutation-run manifest names a live control`;
  `synthetic control labels are pairwise distinct` (mutation: a duplicated label in
  `quantified_helper_controls`, `20260905T010417Z-49456`);
  `every control phrase appears once in the mutation-run manifest` (mutation: one label pasted
  into a second row here, `20260905T011116Z-11226`). Its inventory mutation runs
  misspelled one phrase here and saw both claims report `false` with the missing label and the
  stale phrase named on stderr: a quantified-list row (`value` to `result` in a round-5 row,
  `20260905T004841Z-76830`) and a family label (`20260905T005706Z-70118`).
