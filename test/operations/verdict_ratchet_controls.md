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
| An unknown forwarded option (`?label:e` with `e` neither `Some` nor `None`) is both `e`'s payload and the default, and a later `None` leaves the default (staging#681 round 1, a finding the control rebuts: the path was already so) | `Unknown` counted among the supplied labels, dropping the tagged default | `keeps a wrapper's default alive through an unknown forwarded option` | `20260910T105029Z-41307` |
| A quantifier function is a value only once every population and, but for is_empty, its predicate have arrived; until then it stays a closure carrying what it received (staging#681 round 1) | A quantifier completes on its first positional argument | `refuses a quantifier completed by a later predicate argument`; `refuses a for_all2_exn completed one population at a time` (`does not treat a partially applied quantifier as a Boolean` stays accepted) | `20260910T105036Z-46539` |
| A projection out of an aggregate -- a record field, `fst`/`snd` -- conservatively carries the whole aggregate's sources, as a destructuring of a bound aggregate does (staging#681 round 1) | `Pexp_field` left to the fallback, which reads the record only for the claims it fires | `refuses a quantified field read from a record binding`; `conservatively refuses a sibling field read from a quantified record binding`; the record-field matrix row (`refuses a quantified component read through fst` covers the tuple projections) | `20260910T105042Z-53672` |
| The claim functions live at the top of `Verdict` as well as in `Verdict.Claims`, so an open or a module alias of `Verdict` itself brings them into scope (staging#681 round 2) | Only the `Verdict.Claims` path exports the native claims | `refuses a quantifier written directly in a claim opened from Verdict`; `refuses a quantifier written directly in a claim through a module alias of Verdict` | `20260910T110848Z-84200` |
| A length compared unequal to zero witnesses the population (round 2) | Inequality with zero not read as a witness | `accepts a length inequality with zero as the non-empty witness` (`refuses a length inequality with a positive literal as a witness` stays refused) | `20260910T110854Z-89392` |
| An alternative the value avoided vouches only for the alternatives after it: a later case's guard is never evaluated once an earlier case was taken (round 2) | Avoided alternatives pooled regardless of order | `does not let a later case's guard witness an earlier case's quantifier` (`accepts a witness from an earlier avoided case` stays accepted) | `20260910T110901Z-94684` |
| A format-taking claim whose format is not a literal stays a closure across partial applications, claiming each application's last positional argument (round 2) | The loose closure fires once and returns nothing | `refuses a quantifier reaching a dynamically formatted claim through a partial application` | `20260910T110907Z-99855` |
| A condition selecting a branch that returns a parameter defers the decision -- a `Steering` source -- until the actual argument is known: the constant the branch selects makes the condition the value, the other constant discards it, a parameter defers again (round 2) | Steering sources not created for parameter-valued branches | `refuses a quantified condition selecting constant wrapper arguments`; `defers a quantified condition through a wrapper that forwards its arguments` (`accepts a quantified condition selecting inverted constant wrapper arguments` stays accepted) | `20260910T110913Z-5708` |
| A constructor's payload carries its provenance -- [Ok b], a polymorphic variant -- for the match that reads it out, and a constructor pattern aligns exactly against the same constructor (staging#681 round 3) | Payloads other than `Some` discarded | `refuses a quantifier carried in a constructor payload and matched out` (`accepts a negated quantifier carried in a constructor payload` and `reads a constructor pattern's payload exactly against a matching constructor` stay accepted) | `20260910T121157Z-42194` |
| A builtin -- `not`, `fst`/`snd`, `&&`/`\|\|`, the comparisons, `Fn.id` -- is read as itself only where nothing in scope has rebound its name (round 3) | Builtin rewrites applied before the environment is consulted | `does not read a locally bound not as the Boolean primitive`; `does not read a locally bound fst as a projection` | `20260910T121204Z-47425` |
| A formal's population is substituted only from a component the pattern aligned exactly; a bound aggregate hands every formal the whole argument, which names no one formal's population (round 3) | Population substitution derived from inexact parts too | `does not let a bound aggregate's witness cover a sibling formal` (`accepts a literal tuple argument whose witness and quantifier share the actual` stays accepted) | `20260910T121210Z-52593` |
| A function selected by control flow -- a native claim chosen by an `if`, a wrapper chosen by a match -- is every function it could be, and applying it applies each (round 3) | Alternatives' closures dropped | `refuses a quantifier passed to a native claim selected by control flow`; `refuses a quantifier passed to a wrapper selected by a match` | `20260910T121216Z-57773` |
| A polymorphic variant's payload carries its provenance and a variant pattern aligns exactly against the same tag, as ordinary constructors do (staging#681 round 4) | Variant payloads left to the fallback | `refuses a quantifier carried in a polymorphic variant payload and matched out` (`reads a variant pattern's payload exactly against a matching tag` stays accepted) | `20260910T123138Z-96539` |
| The pipeline and application operators are rewritten to plain application only where nothing in scope has rebound them (round 4) | Rewrite applied regardless of a local binding | `does not read a locally bound pipeline operator as the builtin` | `20260910T123145Z-2339` |
| A recursive group's own members are in scope, pending, while each round walks them, so a forward reference resolves on demand in the first round and the second round is for a binding on a cycle to see its partners resolved -- two rounds, where the cap alone had propagated one link per round (round 4) | The round's own bindings left out of their environment | `refuses a quantifier reached through a three-sibling recursive chain` (`refuses a quantifier reached around a recursive cycle` stays refused) | `20260910T123151Z-7602` |
| A guarded Boolean case is still selected by the scrutinee: a case is reachable on the values its pattern admits that no earlier unguarded case took, and one reachable on a single value is selected by the scrutinee at that polarity whatever its guard adds (staging#681 round 5) | Guarded cases skipped when deriving the scrutinee's selection | `refuses a direct quantifier selected by a guarded Boolean case` (`accepts a direct quantifier inverted by a guarded Boolean case` stays accepted) | `20260910T125004Z-85016` |
| A population key is the expression's text plus the scope of every name it mentions, behind a separator no printed source contains, so a filtered view under a rebound predicate is another population and a parameter mark is read from the scopes, never from the text (round 5) | Only the source collection's scope in a filtered view's key | `refuses a filtered population whose predicate was rebound after the witness` (`accepts a filtered population witnessed under the same predicate binding` and `does not mistake a parameter mark spelled inside a filter predicate` stay as they are) | `20260910T125012Z-90207` |
| A quantifier function that captured a population from a parameter has it substituted when the closure is returned, so its later completion ranges over the actual (round 5) | Quantifier partials left unsubstituted | `refuses a quantifier completed after its population passed through a helper` | `20260910T125019Z-95434` |
| A function selected by control flow keeps every alternative, however many (staging#681 round 6; the bound of round 3 is gone -- the cost it guarded against was the recursive group's rounds, capped since) | Alternatives beyond the fourth dropped | `refuses a quantifier passed to the fifth of five alternative claim functions` | `20260910T131841Z-20969` |
| A `length` is a witness only where nothing in scope has rebound its module (round 6) | The length call read without the environment | `does not read a locally bound List.length as a witness` | `20260910T131747Z-54494` |
| A necessary parameter -- the value's own slot, a conjunct -- brings its actual argument's witnesses to the value; one that is only an alternative brings nothing (round 6) | Witnesses inherited from no parameter | `accepts a witness passed to a wrapper as a conjunct of its claim` (`refuses a witness passed to a wrapper as an alternative of its claim` stays refused) | `20260910T131753Z-60134` |
| An application of a parameter is deferred, its arguments walked where they are written, until a call supplies a function for the parameter, which is then applied to them (round 6) | Deferred calls never made | `refuses a quantifier claimed through a claim function passed as an argument`; `refuses a quantifier claimed through a wrapper passed as an argument` | `20260910T131759Z-65307` |
| A `Stdlib.List`/`Stdlib.Array` quantifier takes its predicate first and positionally, and its pairwise form is `for_all2` (round 6) | Stdlib read with Base's argument order | `refuses a quantifier spelled in Stdlib's argument order` (`accepts a negated quantifier spelled in Stdlib's argument order` stays accepted) | `20260910T131806Z-71069` |
| A recursive module's exports enter the environment like an ordinary module's (round 6) | Recursive module exports discarded | `refuses a quantifier passed to a recursive module's claim wrapper` | `20260910T131813Z-76633` |
| A deferred call's result stands in the body as a placeholder that the result replaces once the call is made, so a wrapper applying a parameter for its value returns that value (staging#681 round 7) | The deferred call's result discarded | `refuses a quantifier returned through a function parameter applied for its result` (`accepts a negated quantifier returned through a function parameter applied for its result` stays accepted) | `20260910T133815Z-33184` |
| A predicate's own sources, its parameters sealed, join the quantifier it decides, at each polarity: an element satisfying the predicate vacuously decides the quantifier as an empty population would (round 7) | Predicate sources discarded | `refuses a vacuous quantifier inside the predicate deciding its quantifier` (`accepts a guarded quantifier inside the predicate deciding its quantifier` stays accepted) | `20260910T133821Z-38390` |
| A let operator binds its pattern to the producer's provenance, conservatively as an identity operator would (round 7) | `Pexp_letop` left to the fallback | `refuses a quantifier bound by a let operator` (`accepts a negated quantifier bound by a let operator` stays accepted) | `20260910T133828Z-43682` |
| An alias or open of a `Stdlib` collection module exports its quantifiers under `Stdlib`'s calling convention (staging#681 round 8) | Exports built under Base's convention regardless | `refuses a quantifier spelled in Stdlib's argument order through a module alias` | `20260910T135708Z-93058` |
| A recursive module group is scanned twice, the second time with the first round's exports in scope, so an earlier module resolves a later sibling (round 8) | One round | `refuses a quantifier passed through an earlier recursive module calling a later one` | `20260910T135715Z-98261` |
| A repeatable population expression -- a record field, a qualified value -- is a population by its text and the scope of every name it mentions, as a filtered view is (round 8) | Fields and qualified values given no identity | `accepts a witness over a record field guarding the same field's quantifier`; `accepts a witness over a qualified population guarding its quantifier` (`refuses a witness over one record field against another's quantifier` stays refused) | `20260910T135723Z-4107` |
| A function handed to something the reader does not model fires its claims with its parameters sealed, so a quantifier over a callback's own population is refused (round 8) | Callback closures discarded at unmodelled applications | `refuses a quantifier claimed inside a callback over the callback's own population` (`accepts a guarded quantifier claimed inside a callback over its own population` stays accepted, as does the round-27 control of a callback parameter that shadows its wrapper's) | `20260910T135730Z-9349` |
| An avoided alternative vouches through its steering only where the steering alone decided it -- an `if` branch, an irrefutable case, a Boolean case -- and as the negation of the steering as a whole, one condition of several (round 8) | Every avoided case's guard read as false, each condition separately | `does not let a refutable case's guard witness a later case` (`accepts an irrefutable case's guard as a witness for a later case` stays accepted) | `20260910T135738Z-14602` |
| A predicate selected by control flow contributes every function it could be (round 8) | Only a single-function predicate read | `refuses a vacuous quantifier inside a predicate selected by control flow` | `20260910T135746Z-19835` |
| A predicate's own claims -- over its element parameter -- fire as a released callback's do (staging#681 round 9) | Predicate closures' claims dropped | `refuses a claim fired inside a predicate over its own population` (`accepts a guarded claim fired inside a predicate over its own population` stays accepted) | `20260910T141528Z-9022` |
| A qualified name in a population key carries the scope of the binding it resolves to, so a module rebound between a witness and a quantifier tells the two populations apart (round 9) | Only unqualified names scoped | `refuses a qualified population whose module was rebound after the witness` | `20260910T141536Z-14305` |
| A functor exports its body, its parameter unresolved, and an application of it exports what the functor does (round 9) | Functor forms export nothing | `refuses a quantifier passed to a wrapper exported by an applied functor` | `20260910T141544Z-19544` |
| A file-local module named like a known one shadows it for opens and aliases (round 9) | Known-module exports before local ones | `resolves an opened local module that shadows List to its own members` | `20260910T141628Z-40484` |
| `phys_equal` and `==` select the other operand against a Boolean constant as `=` does (round 9) | Physical equality unrecognized | `refuses a quantifier compared with true through phys_equal` (`accepts a quantifier compared with false through phys_equal` stays accepted) | `20260910T141557Z-29952` |
| `not` as a value -- aliased, passed along -- is the closure that negates its argument (round 9) | `not` resolved to nothing as a value | `refuses a negated exists through an alias of not` (`accepts a negated for_all through an alias of not` stays accepted) | `20260910T141605Z-35176` |
| A file-local functor keeps its body, and an application scans that body with the argument's exports bound to the parameter (staging#681 round 10) | The argument's exports discarded, the parameter unresolved | `refuses a quantifier passed through a functor parameter's claim` | `20260910T144004Z-65349` |
| A parameter that may also be a callable -- an optional parameter's default -- is both deferred and applied, and the value is either result (round 10) | Only the default applied | `refuses a quantifier passed to a supplied callable optional parameter` (`does not fire a callable optional parameter's inert default` stays accepted) | `20260910T143941Z-60060` |
| A projection out of an aggregate may be any callable component, so a record of claim functions keeps them (round 10) | Aggregates drop every closure | `refuses a quantifier passed to a claim function stored in a record field` | `20260910T143804Z-33374` |
| `for_alli` and `existsi` are the indexed spellings of `for_all` and `exists` (round 10) | Indexed forms unregistered | `refuses a vacuous indexed quantifier` (`accepts a positive indexed exists` stays accepted) | `20260910T143811Z-38602` |
| A recursive module group is scanned once per module, so a chain of any length resolves (round 10) | Two rounds | `refuses a quantifier passed through a three-module recursive chain` | `20260910T143817Z-43839` |
| A let operator defined in the file is applied as the syntax desugars it, continuation included; one from elsewhere still binds its pattern as an identity would (round 10) | Every operator read as an identity | `refuses a quantifier claimed by a let operator's own definition`; the round-7 let-operator controls now name the operator (`conservatively binds a let operator defined elsewhere as an identity` stays as it is) | `20260910T143824Z-49080` |
| An avoided alternative vouches for every later alternative where its steering alone decided it, and only for the alternatives repeating its pattern where a later case repeats it (staging#681 round 11) | A repeated pattern's guard read as decisive for every later case | `does not let a repeated pattern's guard witness a differently matched case` (`accepts a repeated pattern's guard as a witness for the case that repeats it` stays accepted) | 20260910T150114Z-21575 |
| A functor bound by a local module binding is registered for its scope as a structure-level one is (round 11) | Expression-level functors unregistered | `refuses a quantifier passed through a functor parameter's claim inside an expression` | 20260910T150120Z-26832 |
| Two Booleans compared are equal when both are true or both false, so each operand's polarities reach the value (round 11) | A comparison of two non-constant operands read as nothing | `refuses two quantifiers compared for equality` (`accepts two quantifiers compared for equality under a witness` stays accepted) | 20260910T150127Z-32061 |
| A functor is a binding of the environment, so a module prefix, an alias or an open qualifies its name as it does a value's, and an application by any path that reaches it scans the body (round 12) | Applications looked up by the functor's bare name | `refuses a quantifier through a functor applied by its qualified path` (`accepts a guarded quantifier through a functor applied by its qualified path` stays accepted) | 20260910T152217Z-68501 |
| A for loop's variable is a binder of its own, so a population mentioning it is not an outer namesake's (round 12) | The loop body walked with the outer scope | `binds a for loop's variable apart from an outer namesake` (`accepts a for loop body's quantifier witnessed on the loop's own variable` stays accepted) | 20260910T152225Z-79460 |
| A field some assignment in the file writes is read afresh at each occurrence, so a witness taken on one reading covers no other (round 12) | Assigned fields read as stable populations | `refuses a witness on a field the file assigns` (`accepts a witness on a field nothing in the file assigns` stays accepted) | 20260910T152233Z-86035 |
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

## The syntax coverage matrix

The hand-written controls each pin one shape a review round found. What none of them can show is
the cross-product nobody wrote, so the ratchet also generates one (gh-ocannl-931): every value form
below, under each quantifier kind (`for_all`, `for_all2_exn`, `is_empty`, negated `exists`) and each
spelling of the quantifier's function (qualified, through a structure-level `open`, through a
module alias, through a local open), in four cases -- the refusal (the quantifier reaching the
claim in its vacuous polarity, which must be refused), the inverted spelling (accepted), the guarded
spelling (accepted), and the shadowed spelling (accepted: a constant intercepts the value where the
form binds a name, or an ignored sibling receives it). The grid prints in the golden, a cell
reading its case letter where the verdict is as the case requires and `!` otherwise, and each row
is one claim, labelled by its value form. The families are the data table `matrix_families` in the
ratchet; adding a form there adds its row here and its claim to the golden.

| Value form | Claim |
|---|---|
| the argument of a native claim | `a native claim's argument: every syntax matrix cell reads as expected` |
| a structure-level binding | `a structure-level binding: every syntax matrix cell reads as expected` |
| a binding local to the argument | `a binding local to the argument: every syntax matrix cell reads as expected` |
| a helper applied to the population | `a helper applied to the population: every syntax matrix cell reads as expected` |
| a helper written with function-case syntax | `a function-case helper: every syntax matrix cell reads as expected` |
| a wrapper's positional parameter | `a wrapper's positional parameter: every syntax matrix cell reads as expected` |
| a wrapper's labelled parameter | `a wrapper's labelled parameter: every syntax matrix cell reads as expected` |
| a wrapper's optional default | `a wrapper's optional default: every syntax matrix cell reads as expected` |
| a destructured (tuple) wrapper parameter | `a destructured wrapper parameter: every syntax matrix cell reads as expected` |
| a match forwarding its scrutinee through a variable pattern | `a match forwarding its scrutinee: every syntax matrix cell reads as expected` |
| a complementary Boolean constructor match | `a Boolean constructor match: every syntax matrix cell reads as expected` |
| an if condition selecting literal branches | `an if condition: every syntax matrix cell reads as expected` |
| a match guard selecting literal results | `a match guard: every syntax matrix cell reads as expected` |
| a protected try body | `a protected try body: every syntax matrix cell reads as expected` |
| a member of a local module, qualified | `a member of a local module: every syntax matrix cell reads as expected` |
| the same member through a structure-level open | `a member reached through open: every syntax matrix cell reads as expected` |
| the same member through a local open | `a member reached through a local open: every syntax matrix cell reads as expected` |
| a claim fired inside a callback of the wrapper | `a claim inside a callback: every syntax matrix cell reads as expected` |
| an immediately invoked function | `an immediately invoked function: every syntax matrix cell reads as expected` |
| a partially applied native claim | `a partially applied native claim: every syntax matrix cell reads as expected` |
| a pipeline into the claim | `a pipeline into the claim: every syntax matrix cell reads as expected` |
| the tail of a sequence | `a sequence's tail: every syntax matrix cell reads as expected` |
| a tuple component destructured at binding | `a tuple component destructured at binding: every syntax matrix cell reads as expected` |
| a field of a record binding | `a field of a record binding: every syntax matrix cell reads as expected` |
| a comparison with true | `a comparison with true: every syntax matrix cell reads as expected` |

## gh-ocannl-968 mutation runs

All runs used tools/mutation-run.sh on the provenance module and the focused verdict_ratchet alias. Each exited 1 on the listed false claims and restored the source byte-for-byte with cmp. Other listed controls stayed green.

| Mechanism | Mutation | Controls | Run |
|---|---|---|---|
| Boolean aliases retain constants and partial arguments | Unshadowed Boolean operator identifiers no longer resolve to closures | Failed: `refuses a quantifier through an aliased Boolean conjunction`, `refuses a quantifier through an aliased Boolean disjunction`, `refuses a quantifier through a partially applied Boolean conjunction`, `refuses a quantifier through a partially applied Boolean disjunction`. Stayed green: `accepts a witnessed quantifier through an aliased Boolean conjunction`, `accepts a shadowed Boolean conjunction alias`, `accepts a witnessed quantifier through an aliased Boolean disjunction`, `accepts a shadowed Boolean disjunction alias`, `accepts an annihilated aliased Boolean conjunction`, `accepts an annihilated aliased Boolean disjunction` | 20260912T202325Z-58660 |
| Curried functors retain each residual expression and captured environment | A nested functor application cannot supply its next parameter | Failed: `refuses a quantifier through a curried functor`. Stayed green: `accepts a witnessed quantifier through a curried functor`, `refuses a quantifier through a partially applied functor`, `accepts a witnessed quantifier through a partially applied functor` | 20260912T202336Z-60784 |
| Deferred calls preserve exact whole-argument population identity | Saved population keys discarded during replay | Failed: `accepts the same population witness through a deferred helper`, `accepts the same population witness through a deferred partial helper`. Stayed green: `refuses a different population witness through a deferred helper`, `refuses a different population witness through a deferred partial helper` | 20260912T202345Z-62925 |
| Boolean ordering uses both operand views after length witnesses | All four Boolean ordering cases disabled | Failed: `refuses a quantifier in Boolean greater or equal ordering`, `refuses a quantifier in Boolean greater ordering`, `refuses a quantifier in Boolean less or equal ordering`, `refuses a quantifier in Boolean less ordering`. Stayed green: `accepts a witnessed quantifier in Boolean greater or equal ordering`, `accepts a witnessed quantifier in Boolean greater ordering`, `accepts a witnessed quantifier in Boolean less or equal ordering`, `accepts a witnessed quantifier in Boolean less ordering` | 20260912T202355Z-65492 |
