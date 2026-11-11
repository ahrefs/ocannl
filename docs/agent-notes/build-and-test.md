# Build and test mechanics

The dune/OCaml mechanics behind AGENTS.md's workflow rules, and what CI actually covers.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

AGENTS.md holds the workflow rules; these are the dune/OCaml mechanics behind them, narrow enough
that they earn a lookup rather than always-loaded space. The file is far too long to read whole, so
it is sectioned by topic: `grep -n '^##'` lists the sections with their line numbers, and AGENTS.md
points into them by anchor, as `build-and-test.md#<heading-slug>`. Append a new bullet at the end
of the section it belongs to rather than at the end of the file, and open a section when none fits.
A renamed heading breaks every pointer to it; `agent_notes_structure` reports the ones in AGENTS.md,
in the `tools/*.sh` scripts (gh-ocannl-1111) and in the string literals of the OCaml sources under
`test/`, where scanners print them in their refusals (gh-ocannl-1358).

## Repository scans

- Selected documentation compiles through `@test/operations/runtest-doc_examples` (gh-ocannl-660).
  `docs/executable_examples.md` owns the annotation grammar, proposal-history policy and coverage
  limits; `doc_examples.inventory` records unchecked blocks too. Compilation proves API/type/PPX
  validity, while behavioral claims still owe source or executed-test evidence. Code shown as an
  excerpt of a library file carries `doc-excerpt=<path>` and is held verbatim, up to whitespace, to
  that file (gh-ocannl-1265); a fence captioned `from <file>.ml` must declare it or
  `doc-adapted=<path>`.

- The repo-wide scans share one aggregate, `dune build @test/operations/scans` (the `(name scans)`
  stanza in `test/operations/dune` lists them), which `tools/fmt-check.sh` builds after the formatter
  (gh-ocannl-1321; `OCANNL_TOOL_FMT_CHECK_SCANS=0` skips it). What makes a scan rerun is the text it
  reads, not the code it links, so the trigger for running the aggregate alone mid-change is a kind
  of edit, not a subsystem: a config key, a dune stanza, a printed claim, an agent note, a new script
  or source file, an emitted-text pin, or a reference to a library value or type -- test-only edits
  included, since the scans read test sources, goldens and dune files as text (gh-ocannl-1431). The
  focused aggregates `@metal-codegen` and `@lifecycle` are slices of the Metal-pinned tests and the
  run/start lifecycle legs, not of the scans. For shell changes the second half of the batch is
  `tools/test-harnesses.sh --affected --since BASE` (gh-ocannl-1929): exactly the harness commands
  `affected --harnesses` prints (gh-ocannl-1900; a harness's `--only` selects its legs), under one
  slot and one footer, the command `prepush` closes with and does not run; `tools/test-harnesses.sh`
  alone, the full tier, is CI's. Either form is a test batch like any other, taking its `--cpu` slot
  itself through the probe `test-run.sh` and `fmt-check.sh` share (gh-ocannl-1576), foreground or
  detached (gh-ocannl-1236; The harness, shells and processes).

- `markdown_links` (gh-ocannl-2006) joins `@scans`; its closed dialect is documented in
  `Test_scan.Markdown_link_scan`. Fragments and images are unchecked; single-slash root-relative
  destinations are refused. A new symlinked document must join `aliases` in `markdown_links.ml`,
  since Dune materializes symlinks and its links otherwise resolve from the alias directory.
  Published-site exemptions are exact `(file, target)` pairs, not a blanket `.html` exemption.

- A repository-wide scanning check (`config_dep_completeness`, `env_var_deps`, `cache_dir_ignores`)
  is only as good as the distance between what it asserts and what it claims, and a proxy that
  coincides with the property today reads exactly like the property. The working test is: name a
  change that makes the claim false while the check stays green. Each of these was a real gap, and
  each looked airtight until asked that way — "the `.gitignore` line is present" is not "git ignores
  this name" (last-match-wins, so a later `!` takes the coverage away with the line intact); "the
  literal starts with the prefix" is not "the glob covers it" (a glob segment stops at a separator,
  `Schedule_cache.ensure_dir` does not); "every `~cache_dir` carries the prefix" is not "every cache
  directory does" (a direct `Schedule_cache.store ~dir` creates one too, and `""` disables the cache
  only where `Autotune.tune` reads it that way). Two corollaries. A scan that excludes files by
  filename suffix has built an escape hatch from its own "every source" claim unless the exclusion
  is anchored to the directory that generates them. And where the premise is "this value happens to
  equal that constant", read the constant from where it is defined rather than restating the
  coincidence: an unchecked default is the one name no call site spells.
- **The `scans` family is derived, with two hand-listed members.** `env_var_deps` recognises a
  repo-wide scan by its corpus: a `(rule)` whose deps glob the repository recursively, take its
  source tree, or name a root document. `schedule_refusal_scan` is a `(test)` whose corpus is two
  named library sources (`schedule.ml`, `sketch_families.ml`), so no derivation puts it in the
  family and the stanza lists it by hand (its comment says why). Before that, an unmarked raise
  added to `schedule.ml` passed `tools/fmt-check.sh`'s scans and went red in CI
  (lukstafi/ocannl-staging#1766). A scan reading named library sources is in the same position:
  list it by hand, or widen the derivation to cover that corpus shape.
  The other member, `benchmarks/runtest-capped_fixture_scan`, has globs that stay inside
  `benchmarks/`, so `env_var_deps` cannot derive it. It is listed so every push through
  `tools/fmt-check.sh` checks fixture guards: fixtures are copied from neighbours, and a local
  corpus is no reason to let their startup races return.
- **What reaches the scans is wider than what looks like infrastructure.** Two members of
  `@test/operations/scans` census sources across the tree, tests included, so a test-only edit can
  turn them red with no config key, stanza or new file in the diff. `codegen_text_inventory`
  lists each golden holding emitted text at FILE level, with its backend families and the content
  markers it recognises there, and itemises a test SOURCE's pins fragment by fragment, the string
  literals it asserts on included. So any edit that touches pinned emitted text can move the
  inventory's golden and want a re-promote: a test `.ml` that starts or stops pinning a spelling, a
  golden that starts or stops holding emitted text, or a re-promoted golden whose recognised
  markers change. A spelling change inside a listed golden that leaves its markers alone moves
  nothing there, so a green scan is no spelling-level coverage of goldens. `dead_export_scan`
  counts a test's qualified reference to a library value as a caller, and a test's type path as a
  mention of a library type: a test that starts referencing an exempted zero-reference export (or
  an exempted unmentioned type) makes that exemption stale, which fails until it is deleted, and
  one that drops a value's last reference or a type's last mention makes it a new finding.
  lukstafi/ocannl-staging#1095 lost a CI round to both at once: moving `schedule_mma_matmul` from
  emitted-text matching to the mma census dropped two inventory lines and gave
  `C_syntax.is_tensorized_rendering` its first caller. Neither shows in the diff's file list, so
  the `AGENTS.md` trigger names the edit's effect, as broadly as "touches", rather than its
  location or an enumeration of the ways an inventory line can move. Each review round on
  lukstafi/ocannl-staging#1097 found another kind of edit that wording missed, so since
  gh-ocannl-1321 the trigger is explanation rather than gate: `tools/fmt-check.sh`, the format
  rule's last step before every push, builds `@test/operations/scans` after the formatter, whatever
  the edit was.
- `AGENTS.md` stays under 32 KiB, and `test/operations/agents_md_size` holds it there. Claude Code
  truncates an imported instructions file at that size -- silently, from the end, which is where
  the PR, configuration and syntax sections sit -- and the guide crossed the line one review-fix
  bullet at a time, each rule carrying its mechanism and failure story with it. The September 2026
  trim moved the mechanisms into these notes and left the rules as one sentence plus a pointer,
  which is the shape to keep: a new rule in `AGENTS.md` is a line, and what it rests on goes here.
- When `agents_md_size` prints its stderr `WARNING:` (the band from 31 KiB to the cap; the golden
  does not move), trim before the next rule lands rather than after it crosses. How to trim: find
  the bullets that carry their mechanism or failure story (a "because", a gh-number history, a
  second example) and move that text into the matching note under `docs/agent-notes/`, leaving the
  rule as one sentence plus a `<note>.md#<anchor>` pointer, which `agent_notes_structure` checks
  names a real heading; merge two bullets that state one rule; and drop a rule a scan now enforces
  down to its name and the scan's. Never trim by deleting the Pull Requests, Configuration or Syntax
  sections at the end, which are what truncation would lose first.
- The repository's PROSE is checkable where its structure carries meaning, and the agent notes are
  the case in point: `agent_notes_structure` (gh-ocannl-691) reads `docs/agent-notes.md` and
  `docs/agent-notes/` as structure rather than as text. It holds five things true — no bullet cut
  short (each ends in sentence-terminating punctuation, which is what a merge splice destroys), no
  index hook absent from the file its row links, no table row wrapped across two physical lines
  (that ends the table and renders every row below it as pipe-delimited prose), no file unreachable
  from the index, and no bullet promoted into two files. Three of those were review findings on the
  split that created the files, each caught by a human reading carefully. What it costs when you
  append here: end the bullet with punctuation, indent continuations exactly two spaces past their
  bullet, keep one nesting level, keep every index row on one physical line, and give a new file its
  index row the day it appears. It also reads AGENTS.md for its pointers into these files: every
  `<note>.md#<anchor>` there must name a notes file and a heading that file has, since a rule whose
  pointer lands nowhere has lost its mechanism (gh-ocannl-1044). The same holds of the `tools/*.sh`
  scripts and of every string literal of an OCaml source under `test/`, read as the compiler
  decodes it, so a refusal continued across source lines is read whole (gh-ocannl-1358). Comments
  and docstrings there are not read; the scan's own case file is exempt by name, and the exemption
  goes stale the day that file stops yielding a finding. A scanner that prints a pointer therefore
  needs no claim of its own resolving it.
- A scan over a tree that is USUALLY CLEAN needs synthetic negative controls more than one over a
  tree full of findings does, because green-because-intact and green-because-blind are the same
  output. So the rules live in `test/support/agent_notes_scan.ml` as pure functions over strings and
  `agent_notes_scan_cases` runs each of them on a violation it must flag AND on the nearest
  legitimate text it must not — a pipe inside a code span, a bullet ending in `.)` or in bold, two
  bullets that merely open with the same few words. The second half is not decoration: a rule that
  fires on ordinary writing gets switched off rather than obeyed, so every rule owes a demonstration
  that it does not.
- A repository scan that parses generated OCaml must derive those outputs from the generator inputs,
  not name today's targets. `atomic_file_rename_scan` maps the recursively globbed `.mll`/`.mly`
  inputs into a `%{read-lines:...}` dynamic-dependency manifest; its committed ocamllex fixture
  carries a forbidden rename that must arrive through that derived set, so either a blind derivation
  or disconnected scan wiring fails permanently (gh-ocannl-862).
- Public optional arguments in `lib/` follow the caller-visible underscore policy enforced by
  `optional_arg_inventory`: a discard-only value is exposed as `?_feature`, while an implemented
  value uses `?feature`; both mismatches fail. `Optional_arg_scan` parses the source and distinguishes
  real uses from `let _ = feature` / an unused `let _unused = feature` / calls to the unshadowed
  standard `ignore`, including later and nested optional defaults, destructured option patterns,
  functions exported through tuple destructuring, optional closures returned directly or through
  result-position control flow, sequential top-level and local-module bindings, and the identifiers
  that exact unqualified structure- or expression-level `%op`/`%cd` einsum operands (plus `%op`'s
  concat) with the PPX's `=>` dispatch guard turn into generated coefficient /
  legacy-`use_padding` reads under the same lexical scope.
  `optional_arg_scan_cases.expected` ratchets the violating discard forms plus their nearest honest
  counterparts so losing a control is itself a golden change (gh-ocannl-811). Optimizer
  forwarders still need executed oracles — syntactic use proves only that the value was forwarded.

- Ambient gates in `env_var_deps` (gh-ocannl-920) are uncached actions running a program that
  directly declares `arrayjit.utils` and an effective `(link_flags -linkall)`, resolved through
  `Dune_stanza_scan.program_runners` (its identity rules pinned directly by `dune_scan_cases`'
  program-runners table, gh-ocannl-1501) plus Dune's self-running test actions. The force-link flag is required because OCaml can omit an
  unused archive and its startup reader. Grouping and subtraction are evaluated; unresolved
  flag includes, expansions and subtraction of `:standard` are refused explicitly. `(universe)`
  in the action's `deps` prevents caching; preprocessing dependencies do not. It alone identifies
  no gate: compiler censuses and OCANNL-free canaries use it too. The same classification owns alias reachability, the gate's
  training lock obligation and the deliberate generated-alias collision exemption. A mixed Dune
  file exempts only its configuration-free aliases, so a new configuration reader cannot inherit
  the canary's exemption. Each scoped alias must remain live; removing one retires its exemption
  even if another exempt alias still runs. Every action reachable from a scoped canary alias must
  directly run its one local `metal_queue_probe` owner, with only the declared probe module and
  `metal`, `ctypes`, `unix` libraries; alias-only aggregation is allowed. Opaque launchers,
  unresolvable alias dependencies and a reader added to the same alias require a gate.

- A scan resolving names in OCaml sources uses `Test_scan.Lexical_scope` and states its fail-loud
  rule in its header before its first review (gh-ocannl-1140). File-wide literal tables and module
  alias sets cannot distinguish a parameter or nested binding from a same-named earlier value;
  controls must exercise shadowing and separate scopes, alongside the nearest honest use.
- **A scan resolves every name its classification TRUSTS, not only the bindings it tracks**
  (gh-ocannl-1506). `mutation_serial_scan` resolved its tracked tables through `Lexical_scope` but
  first credited `note_mutation` by spelling, classified `Hash_set.add` by its path and keyed
  fingerprinted refs by string name, so a local `let note_mutation () = ()`, a local `module
  Hash_set` or two shadowing refs passed; every one of staging#1367's four review rounds found more
  of that one class. Before the first push, list every function, module, operator (a shadowed
  `not` or `&&`, gh-ocannl-1491) and record field the verdict rests on, and resolve each the way a
  tracked variable is resolved.
- **A scan that DROPS edges to sanction something can pass wrongly** (gh-ocannl-1491). A reach
  check whose edges only accumulate errs toward refusal: what it misreads, it refuses. A sanction
  that cuts an edge errs the other way, so every spelling its recognizer misreads is a silent pass,
  and review keeps finding another. `limits_field_classes` cut `tune`'s reach at
  `seeded_transform`; narrowing the cut to the search-less fallback took seven rounds
  (staging#1362: a wrapper, a bare reference read as a call, a decoy flag, a buried call, a
  shadowed `not`, a rebinding alias). Prefer pins that only add edges or refusals; where a cut is
  unavoidable, state in the first push the one direction it pins and keep its recognizer a closed
  list (Parsing sources in a scan). Pin what the cut trusts the same way: gh-ocannl-1570 bounded
  that fallback's callers by adding an edge back to the route from every caller the closed forms
  do not cover. What syntax cannot decide (a path that tests no flag yet does not search) becomes
  an in-place marker with a reason on the call, and a marker that sanctions nothing adds the edge
  too.
- **For a D1 issue, document the list; add a scan only if it reuses an existing model UNCHANGED**
  (gh-ocannl-1535). A new reader of OCaml name resolution is open-ended, and each review round
  finds a form it misses: staging#1366's process-lifetime census widened `top_level_key_reach`
  (recursive modules, extension payloads, cross-library reach) and took seven rounds, the last
  deferring three forms (`open struct`, a module alias, `module L = Lazy`) no censused source
  contains. The list in the notes is what the small issue owes. Where the list restates a set the
  code owns, it is the restatement Pinning the relationship warns against: the D1 files
  the pin as a follow-up issue, sized for the new model it needs, rather than leaving the list
  unpinned or building that model inside the small issue.

### Bringing the base in

- GitHub builds a pull request's MERGE COMMIT, so a repository-wide scan that is green on your
  branch is not evidence about the tree CI will scan. `agent_notes_structure` (gh-ocannl-691,
  staging#413) survived nine review rounds, `dune build @check`, its targeted aliases and a
  deliberately corrupted copy of the notes, and the merge gate then refused it on nine defects in
  `docs/agent-notes/scheduling-and-autotune.md` — a correctly written note that landed on master
  while the PR was in review, so it existed only in the merged tree (187 bullets against the
  branch's 179). Nothing runnable locally sees that tree unless you bring the base in yourself. This
  matters more for a scan than for an ordinary test, and the difference is the blast radius: a unit
  test fails when its own subject regresses, and its subject is in the diff, whereas a scan fails
  when anyone, anywhere, writes something it did not anticipate — and the population it scans keeps
  growing under it for as long as the PR is open. So fetch the STAGING remote — whose name is local
  and need not be `origin` (AGENTS.md, Pull Requests) — rebase onto its `master`, or merge it in
  where the branch is shared and rewriting is not yours to do (as staging#413 did), and re-run the scan
  BEFORE opening such a PR; where neither is welcome, build the merge commit on a scratch branch
  and run it there. What the omission buys is a false failure on a colleague's correct work, which
  is the outcome that gets a check disabled rather than fixed.
- A negative control written FROM the corpus can encode the ABSENCE of a shape rather than a rule
  about it, and that is the more expensive half of the same story. staging#413's fixtures came from a
  survey of the notes as they stood; the survey found no bullet continued after a blank line, so
  that shape reached no fixture, and the control that came nearest — `- A finished fact.`, a blank
  line, then an indented line — asserted a finding, because a finding is what the scanner gave. In
  Markdown that is a second PARAGRAPH of the same bullet and is correct, which is precisely the
  writing the merge commit failed. A control that asserts the buggy reading is worse than no control
  at all: it makes the wrong behaviour look deliberate, and the next reviewer reads it as the
  specification. So write controls from the RULE, not from the tree — state what the format admits,
  SYNTHESIZE the violating text and the nearest legitimate text beside it, and check the reading
  against the format rather than against what the code answers today. A shape the corpus does not
  contain is the one most in need of a fixture, since nothing in the tree will contradict the
  implementation's guess about it; a survey reporting zero of something is a hole in the fixtures,
  not permission to leave that case undefined.
- Bring the base in before opening the PR; merging does NOT repeat the exercise. Under the
  roll-forward policy (gh-ocannl-861) the gate is one green full-matrix run for the PR's current
  head, and a clean merge proceeds on it however far the base has moved — re-verifying after every
  sibling merge is exactly the cost the policy removed (staging#533 ran three clean rebases and
  three full CI cycles over an unchanged topic diff before it). The gate reads whatever the head is:
  a merge that adds no commit to the branch restarts nothing, while any commit that moves the head —
  a conflict resolution, a rebase, a merge of the base — waits for its own green run, conflicts or
  not. Documentation edits run CI too: selected snippets are compiler inputs (gh-ocannl-660),
  so an absent check is not a documentation exemption. So **a green head merges as it is**: the
  `--merge` merge commit combines it with the newer base, and the PR's run already built
  `refs/pull/N/merge`, the head merged with the base as it stood then. An overlap between the base's
  advance and the PR's own files is information, not a reason to rebase — ship-pr's `merge` prints
  it; read those files for semantic drift, then merge. Rebase, or merge the base in where the branch
  is shared, only for a textual conflict: GitHub reports the PR as CONFLICTING, or the merge is
  refused as not mergeable. Push that resolution only once the head's own run has finished, never
  while it is in flight: `ci.yml`'s `cancel-in-progress` cancels the running one, and its verdict on
  the change with it. Interactions with siblings merged meanwhile are a clean GitHub merge's risk as
  much as a stale branch's. Past the merge build (next bullet) they belong to the integration loop
  on the merged tip or to the CI-red triage routine (the CI section below), not to the session that
  merged.
- The merge itself builds the interactions that are cheap to catch. Two green PRs broke master
  together twice on 2026-10-15: staging#1139 with staging#1142 did not compile, and staging#1171
  with staging#1172 broke `tools/test-test-run.sh`'s repeat leg. `.github/ship-pr-merge-build`
  therefore has ship-pr's `merge` build the PR head merged with master's tip before the merge call,
  whenever ANY path changed on both sides since their merge base, and refuse the merge when
  `tools/merge-build-check.sh` fails there. That script always runs
  `dune build @check @test/operations/scans`, about 100 s from a fresh tree on mac-studio, and adds
  the shell harness tier only when a shared path is a `*.sh`, narrowed to what the shared paths
  reach (the next item). Every path is an input
  because the scans read the repository itself: two notes edits that merge cleanly can still break
  `agent_notes_structure`, push `AGENTS.md` past `agents_md_size`'s cap, or desynchronize a scan
  golden (gh-ocannl-1382). In the month to 2026-10-19, 245 of 474 merges shared a path (166 an OCaml
  input or script, 238 counting Markdown) and 6 a script, so the expensive tier stays rare.
  Run on the two merge commits above, it refused the first in its compile tier and the second in
  its shell tier. A clash across paths only one side changed, or one only a runtime test sees, is
  outside that boundary. To check a merge by hand, run the script at the root of a checkout whose
  HEAD is that merge; `--plan` prints the tiers it selects and why. To size a change to that
  boundary, replay the history rather than rebuilding a script for it:
  `tools/merge-build-replay.py --since '2026-09-19 18:30 +0200' --until '2026-10-19 18:30 +0200'
  --set 'ocaml-sh=*.ml,*.mli,dune,*/dune,dune-project,*.opam,*.sh' --set 'md=@ocaml-sh,*.md'
  --set config --set shell-tier <remote>/master` (the staging repository's remote, whatever
  `git remote -v` names it) reproduces the counts above (166, 238, 245, 6 of 474). Per named
  glob set it prints the merges replayed, the merges that would build, and the paths that
  triggered the ones it adds over the set before it; with no `--set` it replays the config's
  `input` lines and the shell tier's `SHELL_GLOB`, read from where each is defined. Spell a time
  and a zone in `--since`/`--until`: git fills a bare date with the current time of day and a
  zoneless one with the local zone. `tools/test-merge-build-replay.sh` pins its shell-tier replay
  to the script's own `--plan` merge by merge.
- The merge build's shell tier runs what the overlap reaches, and its bound holds the whole tier
  (gh-ocannl-1994). Sized on 2026-11-07 on mac-studio, under a `--cpu` slot from a fresh clone at
  the tier's `--jobs 4` (`SHELL_JOBS` in `tools/merge-build-check.sh`, `tools/test-harnesses.sh
  --jobs`, gh-ocannl-2002), `tools/merge-build-check.sh --all` took 404 s and 403 s: compile 105 s
  and 107 s, shell 299 s and 295 s. Shared-box numbers, at load about 5 peaking at 14 and 17.
  `timeout` in `.github/ship-pr-merge-build` is 1200 s, three times that, sized from what load
  does. `merge` records each build's duration in
  `~/.local/state/ship-pr/merge-builds/<owner>--<name>.jsonl`; the wave before, all 26 were the
  compile tier alone and ran from 99 s to 204 s, and a third `--all` run with `DUNE_CACHE=disabled`
  still compiled in 102 s (396 s whole), so that spread is the box's and not a cold tree's. Twice
  these runs leaves half as much again, the slowest pass at a third of the bound where ship-pr's
  overrun refusal calls four fifths tight. The width is what buys the room: one member at a time
  the same tier took 1071 s and 1056 s, a whole run 1167 s and 1154 s at load 3, and one at load 3.6
  overran the 1200 s of that day, a bound barely over a measured pass. Re-measure before trusting a figure older than the manifest:
  the tier's cost has tripled since one read 470 s. Tier pass lines omit elapsed times so shared
  failures compare without timing noise (gh-ocannl-1950); to recover the per-tier split, time
  `dune build --root . @check @test/operations/scans` and
  `tools/test-harnesses.sh --shell --jobs 4` separately under the same reservation.
  When the trigger holds, the tier is
  `tools/test-harnesses.sh --shell --affected --files PATH...` over every path changed on both sides,
  both names of a rename, non-scripts too, so `affected --harnesses`'s mapping is asked and not
  copied. `--shell` keeps only that group's members: a member of another group would need a
  toolchain the tier does not promise and time the bound was not sized for, so a scoped run is
  never more than the whole tier. The trigger itself stays `*.sh`, a shared non-script alone
  building no shell tier, so `tools/merge-build-replay.py`'s shell-tier counts stand. The fallback
  is fail-closed: a failed ask, a line `--affected`'s allowlist refuses, a path reading as an
  option, a scoped run refused before a member starts (exit 2), and a shared `*.sh` no member of
  the tier reaches, each asked alone so one reached path cannot hide another, all run the whole
  tier. What the mapping answers is the policy: `scripts/harness-support.sh`, which every member
  loads, reaches the whole tier, and a manifest edit reaches the members naming
  `tools/test-harnesses.sh` (three, about 30 s), which is why `tools/test-test-run.sh` compares the
  closing line of `prepush` without spelling that name. A harness named only in another's literal
  buys the slow member every edit of the named file; `tools/test-merge-build-check.sh` pins both
  answers. The nested slot needs nothing: `--list` takes none, and a run inside ship-pr's slot reuses
  it. The base-alone build `merge` compares a failure with carries the merge's two parents on the
  base's tree, so it selects and narrows the same way. A reach the two trees answer differently
  shows as `+` lines, a refusal, and a base red no leg of the scope reaches lets the merge through
  rather than asking for `--override`.

### Scan corpora, goldens and floors

- Repository scans derive source membership through `Test_scan.Source_inventory` (gh-ocannl-871):
  the Dune rule declares `(sandbox always)` plus `(source_tree ../..)`, then calls
  `of_dune_sandbox ~workspace_root ~generated` with its executable, redirected target, and copied
  config in `generated`. `files` returns stable `{ path; on_disk }` entries; `select` and `mem` are
  the corpus API. The clean sandbox excludes VCS metadata and stale build outputs, while
  the explicit generated set removes action occupants; no source list or path encoding crosses argv.
- Configuration consumers outside typed config call sites are covered by `config_usage_scan`
  (gh-ocannl-790). Its source corpus selects every checked-in `*.sh` and `*.py` from that inventory,
  so a new script root needs no allowlist edit; the existing user-help and implementation
  `*.ml`/`*.mli` scope joins it (generated `*.pp.ml`/`*.pp.mli` renderings are excluded). Its prose
  corpus remains AGENTS, the root README, skill, docs, and benchmark Markdown; all checked-in
  `dune` and `ocannl_config` files join too, as do `ocannl_config.for_debug` and both prefixed tokens and
  whole-span inline assignments in `ocannl_config.reference`; workflow YAML under
  `.github/workflows/` contributes unambiguous prefixed tokens. A script or Dune-action
  token contributes a key when it has a qualified command-line spelling and
  value separator accepted by `Utils.cmdline_var_prefixes`, or the environment form
  `OCANNL_<KEY>` beginning and ending at identifier boundaries (with or without an assignment); the
  explicit open namespaces `OCANNL_TOOL_*` and `OCANNL_LOG_LEVEL_<MODULE>` are not runtime config.
  Nor are the cc prelude's C macros: the scan DERIVES them from what `Builtins_cc` defines (the
  dependency-free `arrayjit.builtins_cc` library, so it links no backend) and exempts only their upper-case C spelling
  outside config files; a registered key colliding with one fails (gh-ocannl-1080). The counted
  `non_config_environment_mentions` list keeps only synthetic names no other module owns.
  The explicit token name normally wins over a shorter registered-key prefix. Inherently ambiguous
  alternate-value spellings such as `--ocannl_backend_cuda=true` are file/token/key/count-pinned
  judgments (that example means key `backend` with value cuda=true); any other such ambiguity,
  including a no-equals separator, fails as the longer explicit name. A counted judgment also
  recovers runtime-valid spellings whose key and alternate value separator use different styles,
  such as `--ocannl_print_decimals_precision-7`.
  Supported prefix-free
  config flags occupy the host application's namespace, so their current documentation sites are
  separately file/key/count-pinned and disappear when the runtime's per-key qualified-only policy
  says so. Prose contributes a key
  when an inline code span (including each physical segment of a multiline span) or fenced line
  contains either unambiguous prefixed form; a bare
  assignment contributes only when it occupies the whole inline span outside benchmark reports and
  its unqualified name has lowercase snake-case shape with an underscore; this registry-independent
  grammar comes from `Utils.parse_config_token`, shared with the command-line and environment forms.
  Its structured accepted/ambiguous/rejected result carries explicit names, runtime-shaped shorter
  candidates and normalized diagnostic keys, so this scan resolves registry/site judgments without
  reconstructing the token grammar (gh-ocannl-916).
  Current one-word config assignments such as `profile=reproducible|performance|approximate` and `backend=cc`
  are file/key/count-pinned judgments because their spelling alone cannot distinguish them from an
  arbitrary API or mathematical assignment.
  Whitespace around `=`, within the value, or an empty example value does not hide the key. The
  permanent negative controls are Metal's `fastMathEnabled=false`, Apple's `mathMode=Safe`, and the
  mathematical `d=1`, and tinygrad's `PARALLEL=0`; `debug_log_from_routines=true` pins a real documented positive. An outer bare
  assignment and prefixed tokens embedded in its
  value are both consumers. Config files contribute each uncommented assignment with a nonempty
  value after applying
  `Utils.parse_config_lines`' key normalization: case folding, leading-dash stripping, and the
  optional `ocannl_` prefix. An empty normalized key is retained so registry lookup rejects it. The explicitly included
  `ocannl_config.for_debug` template also contributes commented ready-to-enable assignments. Comment
  markers are recognized against the raw line, matching the runtime parser, so leading whitespace
  does not turn an active invalid key into a silently ignored comment. Because spaced assignments are
  pervasive in code prose, each
  current spaced config mention is itself file/key/count-pinned; newly registered mentions must join
  that list, and a later rename leaves the old pinned mention failing. That boundary leaves fenced
  programs and longer expressions to their own languages. Every config-shaped token is checked
  against `Utils.known_config_keys`; an explicit file/key/count judgment list identifies bare
  assignments that are tensor fields, dimensions, or report notation, and equally narrow counted
  exceptions retain historical invalid spellings and deliberate invalid test controls. Disappearance
  or repetition fails either list, so an unrelated use cannot widen an exemption. The checked-in
  fixture gives every reader form a bogus key and the Dune rule requires the scanner to exit 1 on it,
  so a clean live corpus is not its only evidence that the rule has teeth.
  File-kind non-vacuity claims keep an empty source class loud; Dune's generated local
  `test/operations/ocannl_config` copy is excluded from source membership.
  A test or harness fixture's own environment variable therefore takes a name outside the
  prefix (FAKE_TOKEN, not OCANNL_FAKE_TOKEN), or one under `OCANNL_TOOL_*` when it is harness
  control: any other prefixed name reads as an unregistered key and fails the scan, which cost a
  post-wave worker of 2026-10-06 a review round. The same holds for a name a tool invents for
  itself: a command-line option spelled `--ocannl-<x>` or `--ocannl_<x>`, or a constant or variable
  spelled `OCANNL_<X>`, in a benchmark script, a Python orchestrator or its help text, is OCANNL's
  own config grammar to the scan, a mention of key `<x>`, and fails it when `<x>` is no registered
  key. Name such an option after what it does instead: lukstafi/ocannl-staging#1107's per-cell
  override option for `benchmarks/orchestrate.py`, proposed as --ocannl-flag (unquoted here for the
  same reason OCANNL_FAKE_TOKEN is), failed the scans twice before it became `--cell-flag`. A hit
  in a `*.sh` or `*.py` source now says this in its own refusal and points here; the exemption it
  offers is the environment namespace's alone, as no command-line spelling is exempt
  (`config_usage_scan_script_hint` pins both hints and the unchanged message of an OCaml hit).
- The `.expected` golden of such a repository-wide check should hold what is TRUE of the repository,
  not how much of it there is. A tally — "170 tests in this directory", "241 test stanzas declare
  the config" — moves on every correct addition anywhere, so every unrelated contributor has to
  promote a file they did not touch, and the promote is indistinguishable from blessing a real
  regression in that same file; worse, one hot line in one file collects a textual conflict from
  every parallel branch at once (gh-ocannl-665, where it was an arc's only rebase conflict). The
  counts are usually there as the "the scan did not go blind" signal, which is a real thing to keep
  — so keep it, elsewhere. Three places it can live, all churn-free: PRESENCE in the golden (which
  kinds of stanza a directory has, changing only when it gains its first or loses its last), a floor
  read by a SECOND reader that shares no machinery with the first (`Dune_stanza_scan.raw_stanzas`
  reads the stanzas and what each runs off the raw text, and a sexp walk going blind cannot take it
  down too), and the exact numbers on STDERR, which a `(test)` stanza does not diff. Assert the floor through `Verdict`
  rather than as a golden line, so a scan that goes blind cannot be promoted back to green. A
  total printed BESIDE a list is a tally too, even with the list on stdout: two branches each adding
  an item merge the list right and the total wrong (gh-ocannl-1046 — the refusal-diagnostic and
  config-key counts) — keep the list, send the total to stderr tagged `(not part of the golden)`,
  and give the claim quantifying over that population a `?min` floor. And
  compare the floor to the walk STANZA BY STANZA, never as two totals over a file: the second reader
  recognises fewer shapes than the walk — today an action head nobody has classified — so every
  stanza the walk places and the floor misses is a
  unit of SLACK that can absorb a different stanza silently dropping out of enforcement. Not
  theoretical — the tree stood at 296 placed against a floor of 295, one whole stanza of cover.
  Print the gap ITEMISED rather than as arithmetic: "one short" does not say which stanza is
  standing on the walk alone, and the class it belongs to is what decides whether closing it is
  worth the loss of independence. Once it IS closed, pin the relationship as the two sorted lists of
  stanza IDENTITIES being equal, not as `placed >= floor`: identities say which stanza either reader
  is alone on, both lists move together when a test is added anywhere, and the inequality passes on
  any amount of slack (gh-ocannl-708). Asked per stanza
  the two answers are about the same stanza and cannot be traded against a third, and the narrower
  vocabulary degrades to a weaker floor just where it is narrow instead of to a hole elsewhere. Note
  what this licenses: once the comparison is per stanza, the two readers may share the TRAVERSAL
  that pairs them, because the independence being protected was only ever in the classification. Put the
  independence in the CLASSIFICATION and nowhere else: what can go blind is the checker's own
  traversal and command-recognition, so answer those questions a second way — but PARSE the input
  with the same reader the format admits rather than re-deriving its grammar. Seven review rounds on
  gh-ocannl-665 went into a floor that hand-scanned dune's text, and the same lesson kept arriving
  from new directions: quoted atoms in a `chdir` destination, then in a command, then in a binding's
  value, then in a form's HEAD; whitespace after an open paren; comments where an atom may begin.
  Each was either a false failure or silent under-coverage, and the module's own opening line had
  already said why — an approximation of a grammar has no natural stopping point. Rewriting the
  floor on `Sexplib` dissolved four findings at once and shrank it.
  What the two readers may share when a gap IS worth closing is DATA, never machinery: the pform
  lists (`toolchain_pforms`, and the `%{dep:…}`/`%{bin:…}` prefixes) say which spellings mean what
  and decide nothing, and both readers reading one list is what lets the floor see an external
  command handed a file this workspace builds — `(run python3 %{dep:orchestrate.py})`, whose only
  evidence is in its ARGUMENT. Re-deriving the walk's `classify_command` on the raw side instead
  would make the second opinion a copy of the first, which is the one property a floor exists to
  have. What the raw side then builds on the shared list stays its own and stays COARSER: it may
  under-claim, which weakens the floor for that one stanza, and may never over-claim, which fails a
  correct scan. Resist closing such a gap from the other end by narrowing the walk instead: `(run
  python3 %{dep:x.py})` and `env -C ../sibling ./probe.exe` are the same text, and the second
  launches something of ours somewhere the scan cannot establish (gh-ocannl-708).
  What the second reader still must agree about is SCOPE, and getting it wrong fails correct scans
  rather than blind ones: (a) STANZA POSITION — only top level and inside `subdir`, else `(env (test
  (flags …)))` reads as a test stanza when it names a build profile; (b) WHAT RUNS THINGS — only
  tests, rules and aliases, else a library's `(preprocess (action (run …)))` counts; (c) WHERE it
  runs — `subdir` and `chdir` compose into the directory whose config the process finds, and the
  comparison key is that composition, for a test's own `%{test}` as much as for a helper; (c')
  what CANNOT be resolved — under a `chdir` holding a pform the walk emits a site with no
  executables, so never report a literal `%{…}` as a directory; (d) GROUPING and IDENTITY —
  one site per distinct executable per directory, so raw occurrence counts fail a `progn` running
  one executable twice while a flat set lets five of six rules be answered for by the sixth, and
  identities come from a structured field (`site.executables`), never from splitting a display name.
  And where a floor matches against a CLASS of sites, check that the class is not wider than the
  thing being protected: "any unnameable site" let a `(bash …)` answer for a dropped
  PATH-rewritten one, and the fix was the same as for the executables — give the site a structured
  reason of its own (`site.path_rewritten`) instead of grouping by a shared symptom.
  A last trap in the other direction: declining what you cannot resolve is the SAFE direction for a
  floor and therefore the easy thing to over-use — declining `./%{pp}` left all three `test/ppx`
  rules with no floor at all, and the fix was to resolve the `(:pp pp.exe)` binding from the same
  stanza, which in turn means resolving commands only once the stanza has been read.
  The general form of that trap, and the one worth reaching for first: a shape the second reader
  cannot RESOLVE is usually still one it can COUNT. It does not have to parse a `(bash …)` line, or
  resolve a `(chdir %{…} …)`, to record that the stanza runs SOMETHING — which is the entire
  question a subject-or-not floor asks. Tag such a shape rather than dropping it (gh-ocannl-690),
  and carry the tag WITHOUT the piece that is unknown: a command under an unresolvable `chdir` goes
  into a directory-less list, because the per-directory floors would otherwise hold the walk's
  refusal to guess against the floor's guess. Then watch the OTHER direction, which is where tagging
  puts the risk: a floor that sees a stanza the walk does not fails a correct scan. Mirror the walk's
  own dropping rule at the tag site — a PATH tool is external wherever a `chdir` sends it, so the
  walk places no site and the floor must record nothing.
  Executable IDENTITY has a different origin from that process's working directory: a literal
  `./probe.exe` follows `chdir`, while `%{dep:probe.exe}` (or a named dependency) resolves from its
  stanza. `Dune_stanza_scan.Runs_dependency` preserves that origin, and `command_file_path` is the
  shared resolution used by runner matching; applying the cwd to an expansion a second time hid
  `startup_streams`' bootstrap runner from the ambient census (gh-ocannl-982). Configuration search
  still follows the action cwd, so keep it on the site even for a dependency-expanded program.
  And keep the rule's DECISION somewhere synthetic text can reach it. `env_var_deps`' XOR lived in
  its main loop, so "a rule running its test through `bash` is subject to the rule" could be argued
  and not asserted; `Scan.backend_rule_of` is that decision alone, with the diagnostics and tallies
  left in the check that owns their wording, and `dune_scan_cases` states the rule over stanzas the
  repository does not contain. gh-ocannl-603 gave `config_dep_completeness` the same seam:
  `Config_dep_resolution.analyze` takes a tree as data and returns typed verdicts, the main keeps
  every diagnostic a literal (so the refusal catalogue still finds it), and `dune_scan_cases`
  resolves synthetic trees. Its first run found what no live tree could show: a dune file's
  directory went to `repo_relative` as ONE component, so a `..` popped all of `test/a` and a
  `(chdir ../b …)` there ran in a top-level `b` -- split a directory (`Scan.split_path`) before
  resolving a path against it.
  A source-side census needs the same hostile input: `config_scan_lexing` puts
  `Config_key_scan.generated_init_calls_in_source` to the repository's own traps, since
  `test/support/generated.ml` names `Generated.init` in half a dozen doc comments and
  `generated_provenance.ml` asserts on a string literal quoting one, so a text scan would read the
  module that DEFINES the initializer as its heaviest caller. Two lessons beyond the pattern. First,
  a decision-table control still leaves the wiring from decision to failure unexercised, and
  `env_var_deps --control` closes it: it builds a synthetic tree containing the violating shape,
  hands it to `env_var_deps.exe` as a CHILD, and claims that the child names it and exits 1, and
  that the same tree without it exits 0 — the `generated_provenance` capture pattern, and worth the
  fixture because the tree is DERIVED from the check's own gateless list and so cannot drift from
  it. Second, a vacuity floor over a repository census cannot be asked of an arbitrary tree: the
  family floors apply only when `Config_key_scan.floor_violations` says the run was handed the
  repository's scan roots, and WHICH mode the run was in goes into the golden, so a glob that breaks
  flips that line rather than quietly retiring the floor.
  gh-ocannl-800 adds the corresponding refusal ratchet. `env_var_deps` derives scanner sources from
  the repository-wide rules through the shared `per_directory` traversal, extracts literal formats
  handed to `Verdict.fail`, `failwith`, or a Verdict claim form (including `Printf` formats, with
  substitutions removed), and requires a unique full-format marker plus its readable fragment in the scanner's
  assigned permanent control golden. `Refusal_control_manifest` is the explicit bridge, but not its
  own evidence: a claim marker is emitted only after `Verdict` recorded a passing execution of that
  exact format (and each observation is consumed once), while a direct-failure marker requires
  that the refusal RAN: an `observe_failure` from the caught branch itself, or a
  `raw_direct_evidence` entry naming a control claim that passed AND a `FAIL:` line of a refused
  child run (exit 1) THAT CLAIM made -- `Refusal_control_manifest.refuses ~claim`, or
  `observe_refused_run` over a harness's own capture plus `attribute ~claim` beside the claim
  (env_var_deps' `attribute`) -- that is the refusal's OWN -- matched by its
  format and by no sibling format of the same scanner that fixes more text, so a general refusal
  cannot live on a specific sibling's line. Either half alone counts for nothing: a
  claim over a fixture the scan ACCEPTS, a prose line, or an unrelated passing claim once stood in
  for every direct failure of the two configuration scanners; printed output is no input to the rule
  (gh-ocannl-1207; `Manifest.standing` holds the rule, `refusal_control_scan_cases` pins each
  direction). The claim is a gate -- incidental refusals keep a malformed tree's child at exit 1 on
  their own -- and the line is the evidence; map each key to the claim whose run prints it. A
  line from a sibling claim's run counts for nothing (gh-ocannl-1233): pooled per scanner, the
  runs let a key stand mapped to a claim whose own run never printed it. A key may carry several
  entries, one per claim whose run prints it, and every entry is consulted -- any one answering is
  enough, and a marker that did not print names each of its claims that failed (gh-ocannl-1572:
  a first-match lookup once left a key's later entries dead); the same key and claim twice is a
  copy-paste slip the audit refuses (`Refusal_control_manifest.repeated_pairs`, gh-ocannl-1840).
  A repeated `raw_catalogue_only` key is refused regardless of its reasons (gh-ocannl-1931). A
  refusal no control can reach -- a library constant, a walk defect the raw-text floor exists for
  -- goes in `raw_catalogue_only` with its reason and prints under its own `Catalogue-only:`
  heading, so it stays catalogued without claiming execution; a key in both tables is refused by
  the audit, and a catalogue-only refusal whose own line or caught branch a run shows is refused by
  `print`. The cheap way to execute one is a child run of the scanner over a `with_tree` of
  malformed input (`config_scan_lexing`, `dune_scan_cases`, `verdict_scan_cases`, codegen's
  `--refusal-control`; env_var_deps' `malformed_tree_control` builds its tree beside the shared
  `control_context` through `run_checker`). Every section ends on the unconditional closing row
  `End of synthetic controls.` (`Refusal_section_heads.closing_row`; gh-ocannl-1251,
  gh-ocannl-1318), which is how `tools/mutation-run.sh` tells a complete run from a cut one;
  that row, the marker grammar and the stderr heads below have one owner,
  `test/support/refusal_section_heads.ml`,
  whose source the Perl runner reads at startup by a one-line grammar its header states, failing
  closed (gh-ocannl-1840: they were once held as copies). The marker printer and sync engine
  use that same owner; the runner derives its regex from the verbatim grammar fields (gh-ocannl-1931);
  a marker of the row a run leaves out because a claim behind it failed gets a stderr line naming
  that claim, which the runner lists beside it, and one with none did not run (gh-ocannl-1499).
  Claim markers are consumed as a multiset, so two
  identical claim formats require two executions; refused lines are not consumed, so two sites of
  one direct-failure format would share one line. A new manifest row therefore
  prints nothing until its arm supplies runtime evidence. Never compute a marker by hand: while a
  scan's row is missing or empty, its golden run's `print` writes the whole row, ready to paste into
  `raw_entries`, on stderr (the golden keeps only the section header); once the row is stale -- a
  diagnostic gained, lost or reworded -- it writes the added/removed markers and the replacement row
  there instead, and names the `<source>:<md5>` key of each new direct failure with no entry and
  no in-process `observe_failure` (a separate `--control` run's observations are invisible there),
  whose `raw_direct_evidence` or `raw_catalogue_only` entry stays hand-written -- but offered: once
  the row lists such a failure, its run writes an offer line on stderr naming each passed claim
  whose own refused runs printed it (`Refusal_control_manifest.answering_claims`, the test
  `standing` applies to an entry's claim), or saying none did, and the sync tool prints the entry
  ready to paste (gh-ocannl-1571), never written unasked: which claim answers is a judgment, so the
  reader names it, `--accept-offer KEY=CLAIM` as the offer prints it, and the tool writes exactly
  that entry and re-runs its source's goldens (gh-ocannl-1840). After a rewording,
  one `tools/refusal-manifest-sync.sh` does the rest (gh-ocannl-1429; `--dry-run` shows the plan):
  it rewrites each stale row and reports a paired or dropped old marker outside the current
  grammar as unresolved, leaving its key and golden marker for a hand edit (gh-ocannl-1931).
  For recognized markers, it re-keys the `raw_direct_evidence`/`raw_catalogue_only` entry of a
  reworded direct failure, relabels an evidence entry whose control claim was reworded in its
  scanner's own source (one reworded in another test source changed since `--base`, default HEAD,
  is reported with its new text, never relabeled: the label may back another scanner's evidence),
  swaps the replaced markers and
  reworded claim rows in the goldens holding them, and runs the touched sources' catalogued
  goldens plus the audit, the first run forced so every one prints its offers, then promotes and
  re-runs to a fixpoint (gh-ocannl-1571): the audit reads the scanners' goldens, so its correction
  can surface only after theirs is promoted. It stops on a pass, on a failure with nothing to
  promote (a failed check, said so), or STALE (exit 3) past `--rounds` runs (default 5); a
  correction outside its goldens is named, never promoted (harness:
  `tools/test-refusal-manifest-sync.sh`). It follows a
  diagnostic only where the registered row and the extraction pair it one for one (a rewording
  keeps its place; a rewording beside an insertion pairs nothing), and prints anything else
  `UNRESOLVED` for a hand edit -- then it writes and runs nothing, so a rerun once they are settled
  still sees every source the rewording touched; one old format at two sites reworded differently,
  or text two scanners share, has no single successor and is unresolved too. A source whose
  entries differ from `--base`'s, or a named one, has its goldens re-run. The golden
  swap is not a hand-computed marker: a marker whose refusal the run does not exercise still
  vanishes from the run's output, and the confirming run says so.
  `refusal_control_scan_cases.expected` holds the manifest equal to mechanical extraction, every entry present in the assigned live/case
  golden union, and the manifest's repo-relative source paths equal to `env_var_deps`' derived
  scanner census; the audit's own stanza argument list assigns each source its goldens, and its
  sources are held equal to the manifest's, a missing or extra one named on stderr (gh-ocannl-1088). Rewording a refusal format moves its marker digest: when a row differs, that
  audit writes the markers on each side and the whole replacement row on stderr, and its
  `raw_direct_evidence`/`raw_catalogue_only` claim names on stderr every `<source>:<md5>` key no
  current direct failure answers to -- re-key those from the row difference. The
  audit is itself on `@scans` and excluded from the evidence corpus, so it cannot answer for itself.
  Consequence for authoring: EVERY Verdict claim in a scanner source is a refusal to the ratchet,
  so a claim that is not about the scan's refusals -- a relationship pin between two library values,
  say -- belongs in a non-scanner test (a plain `(test)` stanza the manifest does not list), where it
  costs one golden line rather than a manifest row plus an exercised control (gh-ocannl-719 put the
  approximate ⊇ performance payload claim in `config_profiles`, not `test_config_consistency`).
  The absent-marker, colliding-fragment, one-marker/two-identical-diagnostic,
  one-observation/two-diagnostic, short-literal, `p_all2`, and scanner-population arms prove the
  failure directions. Dynamic strings returned by helpers have no
  scanner-owned literal to extract and stay outside this syntactic contract;
  exact exemptions, when one is deliberate, are stale-checked.
  Four shapes such a scan gets wrong quietly, all found in review, and each is a member of a genre
  rather than a one-off. **Identifiers**: a scan that matches a function through the module it
  belongs to must collect the module's local names from BOTH grammars — OCaml spells binding,
  opening and including twice (`module G = M` / `let module G = M in …`, `open M` /
  `let open M in …`, plus `include M`) — and must consult what it has already recorded, since
  `module H = G` names the module as surely as `module G = Test_utils.Generated` does.
  **Ownership**: a converse check ("this declaration has nothing behind it") must say which stanza
  owns each verdict, or the same fact is reported twice — a rule that runs an executable is judged
  through that executable, one that runs something unnameable is not judged at all, and only a
  stanza that runs nothing whatever answers for its own declaration. **Path identity**: match a
  runner to an executable by the path AS WRITTEN, never by basename, or a rule running
  `../support/probe.exe` credits a local `probe` with a declaration made elsewhere — the same
  collapse the config scanner's duplicate-basename check exists to prevent. **Dune's defaults**: a
  stanza with no `(modules …)` field, or one naming `:standard`, owns the directory less what other
  stanzas claim; reading either as "names no modules" makes a required declaration come out stale.
  **Dune's identities**: an executable is run under `<name>.exe` and under its `public_name` via
  `%{bin:…}`, and a runner named the second way is still its runner. **Subdirectories**: ask the
  question per `(subdir …)` group, not per dune file — the stanzas inside one name modules that live
  there, and a walk that stays at the top level reports a nested source as claimed by nobody.
  **Wrappers over paths**: a signature constraint (`module G : module type of M = M`) wraps the path
  without changing what it names, so unwrap `Pmod_constraint` recursively before resolving an alias.
  **Cross-group relations**: descending into `(subdir …)` is only half of it — a rule outside the
  wrapper runs the executable inside it under the qualified path, so match runners over the whole
  file while resolving modules per group, or descending finds both stanzas and then discards the
  relation between them.
  One more, about the rule rather than the scan: `inline_tests` on a library does not make its
  modules test-only, so a rule that is about what a module DOES when linked (emptying the artifact
  directory, gh-ocannl-723's library refusal) must not accept anything that scopes to the
  inline-test runner alone.
  One more trap that costs a debugging round: dune's `glob_files_rec` runs over the BUILD tree,
  where a `<name>.pp.ml` sits beside every ppx-using `<name>.ml` — and a `.pp.ml` is not input the
  compiler's own parser accepts. A scan that parses everything it is handed (rather than only the
  modules a stanza names) must filter through `Config_key_scan.sources_among` first. Render the floor's answer ALONGSIDE the verdict in such cases
  ("declares neither +floor" versus the same line without it): the pairing is what makes a false
  green visible as a golden line rather than as an absence.
  The
  sibling checks are worth a glance when touching this genre: `env_var_deps` lists
  names only, and `digest_completeness`'s key count moves only alongside its own enumerated key list
  — a number in the same commit as the change it describes costs nothing. Its count of SOURCE files
  was a different matter, and the bullet below is what became of it.
- The same genre had a second instance in the two CONFIGURATION scanners, `test_config_consistency`
  and `digest_completeness` (gh-ocannl-701): their goldens pinned the size of the globbed corpus
  (`Sources scanned: 89 -- arrayjit/lib 47, …`), so every PR adding a source file anywhere under the
  six library directories owed a promote round it had no other reason to make. Note which way round
  the danger ran, because it is the opposite of the noise: two branches adding files in DIFFERENT
  directories write different text on that one line and conflict, which is loud and mechanical,
  while two adding files in the SAME directory write identical text, merge cleanly, and leave the
  total wrong by one for the next unrelated PR to inherit as a red test it did not cause. What
  replaced it is `Config_key_scan.scan_root_floors`, a hand-written lower bound per globbed ROOT,
  asserted through `Verdict` and itemised on failure so the message names the root standing on
  nothing; the goldens keep the root NAMES, since the globs' reach is a fact about the repository
  and a new or vanished root still reads as a diff, and the counts go to stderr.
  Set such a floor well below the day's count and leave it there — the number to raise it to is
  never "today's count", which is the tally coming back in a slower form. Bucket by the configured
  ROOT rather than by a path's own dirname, which is the trap a per-directory tally sets for its
  replacement: these rules glob with `glob_files_rec`, so a source in a subdirectory would otherwise
  open a bucket of its own — a new golden line, which is the promote round being removed, and under
  no floor at all.
- That floor machinery is `Test_scan.Scan_floors`, shared by the configuration scanners and by
  `codegen_text_inventory`: pass it the root table, and get the census-by-root, the itemised
  diagnostic and the stderr report. Each scan keeps its own table, since which roots it globs and
  how far each may fall are facts about that scan.
- The tally rule above is a refusal now, not a review finding: after three hand fixes
  (gh-ocannl-665, gh-ocannl-701, gh-ocannl-1046), `census_total_ratchet` (gh-ocannl-1056) reads
  every golden of the `@scans` family — derived from each directory's `scans` alias, never listed —
  and refuses a run of two or more digits, not continuing a name, that no match of a named `allowed`
  entry spans whole. Entries are floors and caps, fixture locations, text quoted from a source and
  citations, each with its reason, and one that allows nothing is stale; the header states the
  line-shaped boundary. When it refuses a line you added, move the count to stderr tagged
  `(not part of the golden)`; add an entry only for a number that is no quantity of the repository.
  The deliberate exception is `ll_test_ratchet`'s per-source adopted residual counts (gh-ocannl-1090),
  whose growth the golden exists to expose; its anchored allowance covers no corpus-wide total. Its first
  catch was the `FAILED: n checks` teardown total in `config_usage_scan_control`: a negative
  control whose failures ARE its golden ends through `Verdict.exit_negative_control`, which exits 1
  without that line. A one-digit count sits below the number boundary, so the scan also refuses the
  teardown line itself, `FAILED: <n> check(s) did not hold: <names>.` as a whole line, whatever its
  count.

### Pinning the relationship

- PIN THE RELATIONSHIP, NOT THE RESTATEMENT: where a check needs a set that some other part of the
  system owns, relate the two rather than writing the set down again and asserting that the copy
  still says what it says (gh-ocannl-706, after gh-ocannl-591 and gh-ocannl-689 turned out to be the
  same defect twice). Three shapes carry it: a list or a `match` with one entry per member of
  something else's vocabulary; a golden that IS the restatement, printing a list the test itself
  owns; and a count over a derived set. What makes it worse than an unchecked list is the direction
  the drift pushes — gh-ocannl-689's marker vocabulary was closed deliberately, so a stale copy
  rejected the CORRECT marker for a newly added backend and the author's cheapest remedy was `none`,
  a lie the grammar accepts. Prefer DERIVING over pinning where the derivation is free: 29 test
  files copied the backend-name substring test instead of calling the shared predicate, and the fix
  was to call it (the predicate itself became the backend's stated facts, gh-ocannl-1723). Where the list must stay put, four properties make the pin real. Assert it from
  wherever the link cost is ALREADY paid, so no new dependency follows the check around —
  `marker_backend_vocabulary` is a whole executable existing to be the one place that links the
  backend closure. Compare as SORTED LISTS of identities, never as counts or as sets, so a duplicate
  on either side is a mismatch too. Claim one bare boolean through `Verdict`, with both lists on
  stderr, so the golden stays fixed as the underlying set changes and only the literal ever needs
  editing. And give it a control on a SYNTHESIZED violation rather than on the corpus, since a
  corpus where every member satisfies the rule cannot tell a rule from a tautology. The two landed
  exemplars to copy are `test/operations/marker_backend_vocabulary.ml` (gh-ocannl-689, the marker
  vocabulary against `Backends.all_of_backend`; gh-ocannl-706 added the CPU/GPU classification
  beside it) and `config_dep_completeness`'s `Config_dep_resolution.analyze`, which asks the
  question of a tree handed to it as data.
- Not every hand-written list is a member of that habit, and telling them apart is the test of
  whether it is being applied thoughtfully. A JUDGMENT CALL is not a set the system owns anywhere
  else: `digest_completeness`'s `codegen_stage_modules` says which modules read configuration
  downstream of the canonical digest, and `Config_key_scan.scan_root_floors` is hand-written
  precisely so that no scan of ours can move it. Nor is a restatement whose INDEPENDENCE is the
  point — `reduction_forms`'s GPU arms restate their backends' `accum_prec` because a table derived
  from the backend would agree with it by construction and check nothing. A closed vocabulary whose
  unknown word already FAILS is pinned by its own closedness, which is why `config_dep_completeness`
  can print dune's stanza kinds and action heads into its golden. Leave each of those written down
  with its reason next to it, which is what the habit asks of a list that stays.
- The same rule for prose — these notes and `docs/` — lives in
  [conventions.md](conventions.md), in the bullet after the "unfactored duplication" one
  (staging#837): a fact has one owning entry, and the others point there.

### In-place markers

- Where a check needs an EXEMPTION per site, prefer an in-place marker comment to a central list,
  and give it a grammar rigid enough to be wrong out loud (gh-ocannl-659, the XOR between
  `(env_var OCANNL_BACKEND)` and `; ocannl-backend: <word> -- <reason>`). Two reasons, and the
  second is the one that decides it. A central per-site list is the churn and conflict magnet the
  bullet above is about. But the recurrence mechanism is that the next author copies the stanza
  NEXT TO the one they are writing — so the classification has to live there, in the file they are
  already editing, not in a list they will never open. Three things make such a marker checkable
  rather than decorative. (i) ATTRIBUTION BY CONTAINMENT, not adjacency: a comment counts for the
  stanza whose parentheses it sits between, because this repository's dune files habitually leave a
  blank line between a comment block and the stanza it introduces, and "the comment above" would
  have to guess how far above — and would hand a marker to the wrong stanza the first time someone
  left a note between two rules. (ii) A MALFORMED MARKER IS A FAILURE, never a shrug: a grammar that
  silently declines to parse leaves its stanza declaring nothing and reports it as if the author had
  written none, which is the worst of both. So the vocabulary is closed (`none|cc|multidev_cc|cuda|
  hip|metal`), the reason is required and must be more than one word, and the separator is the
  EARLIEST of the spellings admitted rather than the first one that occurs anywhere. The sharper
  form of the same rule, and the one a first draft gets wrong: NEVER NORMALISE WHAT YOU COULD
  REJECT. Filtering empty elements out of a comma list reads `cc,` and `cc,,metal` as a clean
  `cc`/`cc,metal`; deduplicating reads `cc,cc` as `cc`; reading from the earliest sentinel absorbs a
  second declaration on the same line into the first one's reason, where even the accounting check
  below cannot see it, both occurrences being in a comment the scan did place. Each of those is a
  tidy-looking answer to a marker its author got WRONG — and a construct whose entire purpose is to
  be checkable cannot afford to repair its own input. (iii) An
  ACCOUNTING CHECK over the sentinel: every occurrence of `ocannl-backend:` in the file, found by
  the dumbest possible substring scan, must have been read as a marker attributed to a stanza that
  runs something. That is what catches the marker written into a quoted argument, into a field, or
  into a stanza that runs nothing — cases where the author believed they had declared something.
  `Dune_stanza_scan.contained_marker_contract` owns that reusable outer contract (gh-ocannl-863):
  one sentinel per comment, the earliest admitted reason separator, a multi-word reason, raw-text
  sentinel accounting, parenthesis containment, and a convention-supplied wrong-stanza predicate.
  A new Dune comment convention supplies only its declaration parser and subject rule; it does not
  reimplement those refusal classes.
  (iv) READ THE DECLARATION FROM THE FIELD THE ACTION RUNS UNDER, never from the stanza at large. A
  stanza can carry several dependency fields and dune reruns an action under exactly one of them —
  an inline-test library declares under `(inline_tests (deps …))`, and `(preprocessor_deps (env_var
  OCANNL_BACKEND))` in the same stanza reruns nothing that matters while looking, to any
  whole-stanza search, exactly like a declaration. So `site` carries `declares_backend` scoped to
  the same deps field as `declares_config`, and the XOR reads it from the sites; the two answers
  cannot drift because they come from one place. The general form: when a check pairs a claim with
  the thing dune will act on, scope the claim the way dune scopes it.
  The mechanical cost is that comments are what a sexp reader throws away, so the scan needs
  positions: `Dune_stanza_scan.read_raw` returns forms with their byte ranges plus every `;`
  comment, and its tree is compared SHAPE FOR SHAPE against sexplib's, which is strictly stronger
  than the flat form count it replaced and is what keeps a hand-written lexer honest.
- A marker is only as true as the action beside it (gh-ocannl-1518). Every consumer of the backend
  markers -- `Slot_kind`'s width and fleet slot above all -- reads the backends a stanza holds off
  its marker and never analyses its action, so a `cc`-marked rule running `(setenv OCANNL_BACKEND
  hip …)` would take a CPU slot and run HIP at the CPU width. `env_var_deps` therefore asks each
  pinned backend of its stanza's one marker. `Dune_stanza_scan.backend_pins` reads a CLOSED list of
  forms (its header enumerates them): a literal `setenv` of a name addressing the backend, and a
  command-line argument the runtime's own `Utils.cmdline_arg_value "backend"` takes as the backend,
  handed to a program of this workspace -- so no spelling list is restated, and a prefix-free or
  dashed spelling is a pin exactly when OCANNL reads it as one. Anything else in a position where a backend could be fixed
  (an `env` launcher's assignment, a flag handed to a PATH tool, a pform value, a shell line naming
  the key) is refused as unread rather than passed, and an `(env … (env-vars …))` binding is
  refused outright, since it pins stanzas in other dune files no one marker speaks for. The
  decision is `Dune_stanza_scan.backend_pin_issues`, put to synthetic stanzas by `dune_scan_cases`.
- A ratchet whose corpus is the repository's own test sources will scan its OWN fixture, and a
  fixture that pins a shape has to spell that shape out to pin it — so the fixture matches, every
  time, by construction (gh-ocannl-668). Exempt the FILE rather than its matches one at a time: the
  case table grows a row whenever the reader learns a distinction, and a list of its labels
  elsewhere is a second copy of it, maintained by whoever adds the row and read by nobody. What
  keeps that blanket exemption from being a hole is a CANARY — name two of the fixture's literals
  and fail if the scan stops finding them — which is also the churn-free anti-blindness signal for
  this genre, since an empty offender list and an unread corpus are otherwise the same result.
  Spell one canary in a form only the real reader can see (a string literal broken over a line
  continuation, whose decoded value spans no single line of the file): it fails a scan that has
  quietly regressed to matching text, which the plain spelling would not.

### Parsing sources in a scan

- Scan OCaml sources through **ppxlib's** parse tree (`Ppxlib.Parse.implementation`,
  `Ppxlib.Ast_traverse.iter`), never `compiler-libs`'. The compiler's `Parsetree` moves between
  releases, and the breakage lands in the scanner as a compile error rather than in anything it
  scans: 5.5 alone gave `Ldot` located components and stopped spelling `let module M = … in …` as
  `Pexp_letmodule`, making it an ordinary structure item inside the expression — so no single arm
  compiles on both sides of that boundary. ppxlib parses with the compiler's own parser and then
  migrates the tree to an AST of ppxlib's version, which is what decouples a scanner from the
  compiler it is built by. That AST is `Astlib.Ast_502` — BELOW the declared OCaml floor, so every
  parse is a downgrade, and the reassurance is not "nothing newer can arrive" but that the 5.x
  downgrade chain performs no `migration_error`: a construct the older AST spells differently is
  mapped onto the older constructor (5.5's structure-item-in-an-expression becomes
  `Pexp_letmodule`) or carried in an attribute encoding. The corollary is that the coupling moves
  to ppxlib — **`dune-project` bounds ppxlib above** for that reason, since the ppx matches the
  same selected AST at some sixty sites. Match with
  ordinary patterns; metaquot quotations (`[%expr [%e? f] ()]`, `[%expr ()]`) are preferred wherever
  they express the WHOLE shape, since they ignore locations and attributes while keeping arity and
  labels exact. What they cannot reach — a variable-length argument list, a string constant's value,
  a module binding — stays written against the constructors.
- A scan's failure on a source is classified through `Test_scan.Scan_attempt`, never a
  `try … with _ ->` that calls every exception "does not parse": `run` tells the parser's errors
  (the input's fault, `Unparsed` carrying the parser's own one-line report: line, columns, what)
  from any other exception (the scanner's, kept with its one-line `Printexc.to_string` text), and a
  case file's per-snippet catch-all is `attempted ~what ~name ~default`, which reports either as a
  `Verdict.fail`. Each repository scan's two refusals are separate formats, the reader an argument
  of its `*_or_refusal` function so the refusal control can plant a raise and claim that exactly the
  defect message fired (`routed_case`, gh-ocannl-1292); `env_var_deps`' tracing-gate walk reads each
  module of an explicit `(modules …)` that way (`module_reads_or_refusal`, gh-ocannl-1500), where an
  unparsable one used to raise out of the whole check. Not yet routed: the dune-file readers,
  whose parse errors are sexplib's or `Dune_stanza_scan`'s rather than located ppxlib ones.
- Type-check every OCaml snippet a scan's cases feed it before trusting the arm. The parser
  accepts programs the type checker rejects — a duplicate top-level `module` parses cleanly — and
  an arm over such a snippet pins the scanner's reading of code no compiler would build, so its
  verdict says nothing about the tree the scan polices. `tools/typecheck-scan-cases.sh <table>.ml`
  does it: each `{ocaml|…|ocaml}` snippet of the table (the case-table shape; an ordinary `"…"`
  literal is not read) becomes its own unit under `ocamlc -stop-after typing`, reported by case
  name and table line. Stand-in names (`key`, `Autotune.tune`) come from the table's companion
  stub, `test/operations/scan_case_stubs/<table>.mli`, compiled first and OPENED at each arm's
  top, not pasted in: an arm may rebind a stub's name (`module Autotune = Other`), legal over an
  open and a duplicate definition beside a paste. A name an arm uses unbound — a `Cache` after the
  local alias's scope ends — is a stub entry too, saying what a compiling program binds there. An
  arm that feeds a non-program on purpose (what the parser makes of `F(X).x`), or one whose fix is
  a follow-up, is listed in the stub as `[@@@known_rejection ("case", "error text")]` and must stay
  rejected with that text; a library a snippet opens is `-I` its directory of `.cmi` files.
  `--stubbed` runs every table with a stub, and `tools/test-typecheck-scan-cases.sh` (a toolchain
  harness) holds them clean, so a new arm on a stubbed table is checked by the harness tier. Doing
  this for `cache_dir_scan_cases` (gh-ocannl-1248) found five arms no compiler accepts, one of
  which pinned a mis-modelled `module X :=` (a local substitution, not a removal); the script's
  first run found two in `config_scan_lexing`: a partial application destructured as a pair
  (fixed), and an optional argument applied to a bare literal (`?l:"k"`, which never types;
  listed). A snippet in project syntax (`let%op`, `[%cd …]`, `[@@deriving …]`, any extension
  node) needs its stanza's preprocessors, libraries and DSL scope, which bare `ocamlc` lacks: the
  stub names a module of the repository whose stanza such code lives in, and the prelude its arms
  start with, as `[@@@owning_module ("lib/nn_blocks.ml", "open! Base\nopen …DSL_modules")]`.
  The engine builds that module's directory and compiles each such arm with the flags dune
  itself reports for the module (`dune ocaml-merlin`: the `-ppx` driver, the `-open`s, every
  library's `-I`), so the context is the stanza's reading, never a restated copy of its fields
  (gh-ocannl-1587). Doing it found the `let%op` arms of `optional_arg_scan_cases` in a spelling
  no compiler builds: an `%op` function with no `()` parameter then read even an optional
  parameter as a label source (`stride.Tensor.value`; gh-ocannl-1787 exempts optional ones), and
  a `"stride*o+k"` spec reads `use_padding`, so the arms took a `()` and the stub a
  `use_padding`. An extension no preprocessor expands (`[%%ext …]`) is then
  an ordinary rejection, listed where the arm is that parse shape on purpose. A snippet passed as
  `~interface:` is compiled as a signature. The context needs the project's libraries, so the harness's
  leg 1 runs `--no-project-context` (project arms REFUSED) and the `#project` legs check them:
  CI's toolchain job, which lacks those libraries, expects their skips
  (`tools/harness-skips-ci-toolchain.txt`), and the build matrix runs them strict.
- A scan that MODELS library or language semantics (which calls are wrappers, which arguments of a
  library function carry a collection, which bindings a gate reads) states its CLOSED LIST OF
  RECOGNISED FORMS in its reader's header and in the PR body FROM THE FIRST REVIEW ROUND, not once
  review has found the edges. The `Lexical_scope` bullet under Repository scans (gh-ocannl-1140)
  asks the header for the fail-loud rule; this asks it for the membership the rule is relative to,
  every member enumerated down to the arguments it reads, with no trailing "..." standing in for a
  semantic model the reader has not written. Outside the list the reader refuses ("unread") rather
  than passes, which is the In-place markers rule (never normalise what you could reject) applied to
  a reader. Where refusing would hit ordinary valid code, the exclusion is a named decision in the
  list's boundary, not a silent end of the walk. Then classify each review finding against the
  list before touching the reader. A form the list names and the reader misses is a must-fix silent
  defect in any round. A corner outside the list goes to ONE follow-up issue and stays out of the
  reader. Without the list nothing tells the two apart, and every finding becomes a form to add.
  staging#1089 (gh-ocannl-1289, `Verdict_scan.gate_census`) paid for it in 13 remote rounds, and is
  the cautionary case, not the exemplar. Its first body did list the population walk's forms, but
  one member was an open "standard collection module (`List`, `Array`, ...)" model of which
  arguments carry the population, and a form outside the list ended the walk silently. So each
  round found another form a population is built through (Base argument order, writers into
  mutable containers, `ref` initializers, `init` callbacks, ...) and fixed it in place. The body
  called the list closed only from round 7, and its deferred residue is ahrefs/ocannl#1303, which
  brought the reader to this rule: its name-prefix heuristic became `collection_functions`, an
  exact name-to-role table (which arguments build the population, whether the function quantifies,
  whether it writes into a mutable container), and a collection-module function the table does not
  list refuses the claim as `unlisted`, naming the function to add. Outside the list the walk still
  stops at the boundaries its header names with a reason -- above all a call to any other function,
  where refusing would refuse every population a helper function builds.
- `dead_export_scan` (gh-ocannl-806) is a source-level ratchet over direct `.ml` modules without a
  sibling `.mli` in `arrayjit/lib/`, `tensor/` and `lib/` (gh-ocannl-1009): it enumerates source-declared top-level `let` and
  `external` values plus the values and modules generated by top-level `of_sexp`, `sexp`,
  `compare`, `equal`, `hash`, `enumerate`, `fields` and `variants` derivings (gh-ocannl-913: the
  accessors, constructor values and `Fields_of_<t>`/`Variants_of_<t>` modules included; a modelled
  deriver applied to arguments is refused), then counts qualified references from other sources. Source-declared `let`
  names starting with `_` are excluded by policy (gh-ocannl-941), matching OCaml warning 32;
  this reaches patterns and extension payloads, but leaves external and deriving-name census
  policies intact, including polymorphic-variant parser helpers. `[%of_sexp: M.ty]`,
  `[%compare: M.ty]`, `[%equal: M.ty]`, `[%compare.equal: M.ty]`, `[%hash: M.ty]`,
  `[%hash_fold: M.ty]` and `[%all: M.ty]` count as references to the corresponding generated
  value; the walk also reaches type constructors nested inside the extension's type and those
  consumed by another deriving. A call that a deriving of the defining module itself generates
  (`compare_entry` calling `compare_scope`) is a support reference (gh-ocannl-914): it counts
  while its caller is live, so a helper a live sibling needs is not reported and one whose caller
  is dead is. `dead_export_scan_cases` holds every derived name and inferred call equal to the
  installed PPX expansion of a fixture corpus; a fixture whose ignored component shares its type
  with a kept one cannot see a misread ignore attribute, so each one carries its own type.
  `sexp.opaque`, `sexp.ignore`, `compare.ignore`, `equal.ignore` and `hash.ignore` suppress the
  corresponding implicit reference (ppx_hash also skips what `compare.ignore` marks), and an inherited polymorphic-variant row calls the inherited type's `__<type>_of_sexp__` parser helper.
  A polymorphic-variant `sexp` deriving contributes that additional helper to the export set.
  **`sexp_of` converters are excluded by policy** — the `sexp_of` half of a `sexp` deriving as
  much as a standalone one or a hand-written `let sexp_of_x`, recognized by the name prefix, and
  `[%sexp_of: M.ty]` credits nothing. They are the debugging and
  observability entry point, and `ppx_minidebug`'s typed log annotations (`[%log (x : M.ty)]`,
  the parameters of a `%debug_sexp` function) expand to converter calls a source-level census
  cannot see, so a converter used only from a log read as dead; a `sexp_of` with no caller costs
  nothing and cannot drift from its type, and removing it to satisfy the ratchet only takes it
  away from the next debugging session. `of_sexp` stays censused: it is the persistence entry
  point, an unused parser over opaque fields is a runtime trap, and narrowing `sexp` to `sexp_of`
  is a real cleanup. Module aliases are
  followed conservatively; an unqualified identifier in the lexical scope of `open M` counts even
  when shadowing could make it local. An `include M` is NOT a use (gh-ocannl-1085): crediting every
  value of `M` there hid `Datatypes.mutable_list` behind `utils.ml`'s `include Datatypes` for years.
  The including module — a file, or the `module N = struct … end` or `let module N = … in` around
  a nested include — becomes a further receiver of `M`'s values instead, transitively and through
  a module alias of either declared in any source, so `Utils.insert` counts for `Datatypes.insert` and so does a bare `insert` under
  `open Utils` or below the include. Includes of a functor application, or inside a functor body or
  an anonymous structure, are not followed, so a use through one reads dead — the loud direction;
  the other choices admit false positives rather than refusing valid code. Values created
  by other PPX expansions or brought into the defining module by `include` are outside the census. Every current zero-reference export is an exact stale-checked exemption: adding an `.mli`,
  removing the value, or giving it a detected caller requires deleting its exemption.
  Top-level TYPE declarations in the same modules (gh-ocannl-1009) are a coarser census: a type
  whose name appears in no type path anywhere in the tree (`.mli` signatures included) outside its
  own declaration, and in no value or module path spelling what its deriving generates (a
  `[@@deriving]` payload names derivers and credits nothing), is reported. It is a name
  count, never a resolution: labels and constructors — `fields`' label-named accessors
  included — are not resolved to their type, so a type
  built and matched only through them is a false refusal exempted with that reason, while any
  same-named type path elsewhere credits it. Crediting label or constructor names instead fails on
  the motivating record: `Train.example_train_result`'s labels include `inputs` and
  `used_memory`, names the tree spells hundreds of times, so it would have been credited and missed.
- A documentation comment survives into the parse tree as an `[@@@ocaml.doc "…"]` attribute holding
  a STRING, so an iterator hands it to an expression hook exactly like code would (verified by
  removing the guard — the prose cases flip to findings). Any scan over string literals must
  therefore override `method! attribute _ = ()`, or this file's own documentation of the pattern it
  hunts becomes a finding. Ordinary `(* … *)` comments need no such
  care, the parser drops them outright. Extension payloads are the opposite call and must stay
  visited: `[%cd …]`, `[%expect {|…|}]` carry code, or the golden text of some.
- `git` strips TRAILING spaces from a `.gitignore` pattern (unless backslash-quoted) and keeps
  LEADING ones, so an accidentally indented ` /foo/` is a pattern beginning with a space and matches
  nothing. Any code reading that file must not `String.strip` both ends, or it reports coverage git
  does not give; `#` likewise opens a comment only at column 0. A backslash escapes the next
  character, in both directions that matter: `\_` is an underscore, so `!/foo\_bar/` really does
  un-ignore `foo_bar` while a matcher reading the backslash literally sees no match and reports it
  still ignored, and `\*` is a literal asterisk rather than a wildcard, so ignoring the escape
  over-matches as readily as it under-matches. A leading `**/` matches any number of directories
  INCLUDING ZERO, so `**/foo` and `/**/foo` both reach a root-level `foo` and cannot be dismissed as
  "contains a slash, so it is anchored elsewhere"; consecutive asterisks anywhere else are ordinary
  ones. All of this verified against `git check-ignore`
  rather than read off the documentation — which is the cheaper move whenever the question is what
  git actually does.

## The harness, shells and processes

- `tools/test-harnesses.sh` runs the complete hand-run harness tier locally, from any
  working directory. Its manifest is the membership CI uses: the toolchain-free job runs
  `--shell`, the formatting job runs `--toolchain` and `--promotion` after installation, and the
  `promotion-floor` job runs `--promotion` again under the Dune that `dune-project`'s
  `(lang dune X.Y)` declares (gh-ocannl-1215); omitting the group runs all three. Both
  toolchain-installing jobs set `OCANNL_TOOL_HARNESS_STRICT=1`, under which `harness-support.sh`
  reports a skip as a FAIL: a counted skip is right where a host lacks a capability, and a vacuous
  green where the step installed it. setup-ocaml's switch is not on a CI step's PATH, so a harness
  needing `dune` falls back to sourcing `tools/opam-env.sh` before `harness_require dune`; one that
  omits the fallback skips every leg there, which strict mode turns red.
  `--list` lists the selected paths without executing them. Every selected member runs after
  an ordinary failure; the footer counts failed harnesses and exits 1 if any failed. A member
  interrupted by HUP, INT or TERM stops the aggregate and preserves exit 129, 130 or 143. Individual
  harnesses retain their counted capability skips; the formatter's real cache control and
  promotion fixtures need the local toolchain. For a shell change, run the command
  `tools/test-run.sh prepush` closes with, `tools/test-harnesses.sh --affected --since BASE`
  (gh-ocannl-1929, below: the members and legs the change reaches, the whole tier being CI's) beside
  `tools/test-run.sh run build @test/operations/scans`: the latter includes configuration
  usage and shell parsing, which standalone process fixtures cannot cover (gh-ocannl-988).
  `tools/test-test-harnesses.sh` drives an isolated copy with derived member stubs, including
  a fail-fast twin that must lose the all-members oracle. Setup-hook fixtures derive their
  helper files from the hook's source directives and refuse unknown forms; an injected helper
  passes through this derivation and fails against the former fixed-list fixture.
- On a fleet box a harness run is a test batch like any other, and like `tools/test-run.sh` and
  `tools/fmt-check.sh` it takes one of the box's correctness slots itself, as `--cpu`
  (gh-ocannl-1576): after its arguments and skip lists pass, it runs itself again under `fleet-worker.sh execution
  slot --wait <OCANNL_TOOL_SLOT_WAIT, 600> --cpu`; `--list` takes none, a refused slot exits 75
  (`harnesses: FAILED (exit 75): SLOT REFUSED`, no member run), and an enclosing slot is reused
  (the nested-slot rule). Every OCANNL asker -- the three slot-takers, the sweep's registry reader
  and its GPU-token reading on the box -- finds the fleet-worker.sh through ONE probe,
  `tools/fleet-worker-candidates.sh`'s `fleet_slot_probe`, whose header states the answer's grammar
  as a fail-closed boundary, so a change to the probe (its bound, the capability check, the
  answer's fields) lands everywhere at once (gh-ocannl-1576, gh-ocannl-1841). Its bound is a
  caller's hook: the sweep passes its `run_capped`, so a cancellation reaches the ask, and the
  token reading ships the functions to the box as `declare -f` text under the box's bash rather
  than restating the loop in remote shell. The harness runner and `tools/fmt-check.sh` rerun
  themselves in the slot through one helper beside it, `fleet_slot_rerun`, with a caller-named
  environment variable for the record, a caller-given message prefix, and one set of outcome
  lines; before gh-ocannl-1841 each carried a 30-line copy, and the sweep's probe parsed the line
  more loosely than the token reader did. `tools/test-test-harnesses.sh` drives the slot with a
  fake fleet-worker.sh (one answering no probe is the negative control) and checks that the askers
  call the shared functions and no tool asks `execution slot --probe` itself (red against the
  pre-gh-ocannl-1841 copies in all four files).
  It runs in the foreground or detached. Under `bg-run.sh spawn`, `execution slot --bg` or
  `nohup` it inherits SIGHUP ignored (and a background `&` SIGINT), which a non-interactive bash cannot trap: before gh-ocannl-1236
  `tools/test-test-run.sh`'s HUP lifecycle legs failed there, on master too, and the harness
  aborted before its later legs (three workers of the 2026-10-05 wave and two of its post-waves
  paid a base comparison or a rerun). Now the signal legs launch their fixture through a perl
  that ignores INT, HUP and TERM, standing for such a caller in EVERY run, and a second that
  resets them, so the reset is what each run exercises and the legs pass whatever the harness
  inherited (its header line `inherited ignored signals:` says what that was); a lifecycle
  fixture that never becomes ready fails and ends its leg, never the harness. A launcher that
  keeps HUP ignored on purpose (`nohup`) is right to: the fix belongs in the fixture, not in
  the callers.
- The harness of `tools/test-run.sh` is six members over one shared file (gh-ocannl-2002). As one
  file, `tools/test-test-run.sh` was some 70 legs and 764 s on macbook-pro (735 s on mac-studio), 70%
  of the shell tier, and every path its text named reached all of it. The leg clock showed no
  leg to cut (`OCANNL_TOOL_HARNESS_LEG_TIMES=FILE`, `scripts/harness-support.sh`: every gate
  appends `SECONDS<TAB>HARNESS<TAB>LEG` for the stretch that ended, charged to the gate that
  opened it): the slowest four were 71, 67, 65 and 62 s, in four groups, and half of the whole
  is fixtures waiting (371 CPU-seconds). So the split buys reach and time, not fewer checks.
  Reach: an edit of a member (`tools/test-test-run-NAME.sh`: `options`, `digest`, `devices`,
  `lifecycle`, `last`, `bisect`, 86 to 165 s each) reaches that member, where any edit of the
  one file reached all 764 s; `tools/test-run.sh` and `tools/test-test-run-common.sh` still
  reach all six, since every leg runs the script. Time: `tools/test-harnesses.sh --jobs N` runs
  up to N members at once, each one's output kept and printed whole in the manifest's order, so
  the six take 165 s beside one another and the whole shell tier 301 s at `--jobs 4` against
  1066 s one at a time (macbook-pro, a shared box at load 2 to 25; shared-box numbers, not a
  measurement). CI's `harnesses` job and the merge build pass `--jobs 4`; the default stays one
  at a time, live. What a member shares is `tools/test-test-run-common.sh`, sourced: the
  options, the hermetic environment, the subject, and the fixtures legs of more than one member
  read, as STAGES. `shared_through NAME` ahead of a leg runs every stage not yet run up to NAME,
  in the one order the legs were written against, so a leg starts from what it started from in
  the one file: the split moved no check (the PASS and SKIP lines of the six are those of the
  one file, plus each member's own extraction lines and closing leg). A fixture only one member
  reads sits in that member; one a second member comes to need moves to the common file as a
  stage. Leg numbers are the harness's, unchanged, so a member's are not consecutive; leg 70
  (no device probe leaks from a leg that faked one) closes every member. `tools/test-test-run.sh`
  is now the one command for a person: it runs every member, or with `--only` the members
  whose `--list-legs` shows a leg the selector matches, so `--only '^81:'` needs no member's
  name (leg 105 pins it); it takes no slot and is no member of the tier.
- Real-dune harnesses matching diagnostics or promotion listings use
  `harness_dune_colour` from `scripts/harness-support.sh` at the fixture command,
  so setup-ocaml's `CLICOLOR_FORCE=1` presentation is exercised locally too
  (gh-ocannl-2070). `harness_plain` strips ANSI at the matching boundary while
  retaining the raw log. The promotion, digest and scan-case harnesses pin
  coloured fixtures; the formatter subject, ppx target-listing subject, promotion
  subject, affected-test queries and saved-promotion recovery run with colour
  forced. Alias validation already tests both colour settings.
- A harness names its locale (gh-ocannl-2002): `harness_args` exports `LANG=C` where neither
  `LANG` nor `LC_ALL` is set (a headless ssh session). There a Homebrew bash asks CoreFoundation
  for the user's preferred languages whenever a script assigns a locale variable (`local
  LC_ALL=C`), and that lookup segfaults when several processes make it at once: on macOS 27 with
  bash 5.3.9, 12 of 40 runs of `tools/test-ci-durations.sh` beside seven copies of itself, none
  of 40 with `LANG=C`; one such crash reddened the first `--jobs 4` run of the tier. It shows as
  a subject exiting 139 with nothing on stderr, and
  `~/Library/Logs/DiagnosticReports/bash-*.ips` names `libintl_setlocale`.
- Selecting legs (gh-ocannl-1202): a harness built on `harness_leg` takes `--only REGEX` over its
  leg names, and the members of `tools/test-test-run.sh` close each name with its one group, so
  `--only '#lifecycle'` runs the run/start lifecycle legs with the signal legs and both fork
  negative controls (leg 74's unmasked parent-ownership and inherited-handler twins); the groups
  are listed in each member's header, and a group can span two (`#lifecycle`: leg 99 sits in
  `last`), which `tools/test-test-run.sh --only` runs both of. Setup and cleanup outside the gates run as in a full run; a selector
  matching no leg exits 2; a filtered run lists the legs it ran and each check they skipped,
  with its reason, above a `FILTERED RUN, not a full pass` footer. To sample a flaky leg, loop
  over the selected run and count its `FAIL  ` lines.
- Testing another copy of the script (gh-ocannl-1925): `tools/test-test-run.sh --subject FILE`
  (or `OCANNL_TOOL_HARNESS_SUBJECT=FILE`; the option wins) runs the legs against FILE in place of
  `tools/test-run.sh`, so "this leg fails against the previous script" is
  `git show origin/master:tools/test-run.sh >F` and `--subject F --only '^N:'`, expecting exit 1 —
  not a scratch copy of `tools/` and `scripts/` with the script swapped in, which the
  gh-ocannl-1830 worker made about ten times. FILE is staged once as `tools/test-run.sh` of a root
  under the scratch directory beside the files it sources, and every fixture root, mutant and
  direct call reads that copy. Only the script is swapped: what it sources, the shared
  `scripts/`, and the harness itself stay the working tree's, so a control for a change that also
  edits a sourced `tools/*.sh` still needs that file put back by hand, and a leg that reads the
  script's TEXT (a mutant's seam, an extraction) abandons itself where the older FILE lacks the
  line — a FAIL that says so, not the leg's verdict on FILE. Leg 102 pins the option with a
  subject missing one `proc_table_perl` rule: leg 99 fails in the one check that rule answers.
- **Every gated leg must pass run alone** (gh-ocannl-1367). A leg that reads what an EARLIER
  leg's body made (a variable, a file, a lock, an exported mode) passes in every full run and fails
  only under `--only`, as two did when `tools/test-test-run.sh` was first gated (`LAB_MAP`,
  `slot_dirs`). Build what a leg needs inside it (a module, a probe mode, a lock it reads), or as a
  shared fixture outside every gate, and define a helper two legs call outside both. `tools/harness-legs-alone.sh [HARNESS...]` runs each
  harness once in full, then every leg and every `#group` alone, reporting `ALONE FAIL` (with
  whether the full run passed), `ALONE SKIP` (a check skipped alone that the full run decided) and
  `ALONE MISS` (the selection did not run exactly the legs meant), and `CHECKOUT CHANGED` for any run
  that leaves a tracked or untracked file behind (a later run could pass on it); with no argument it takes every
  member of `tools/test-harnesses.sh` whose gates list a leg. CI runs it on the daily schedule and
  on a dispatch (job `harness-legs-alone`), since it costs a run per leg; run it locally, slot-wrapped,
  on a harness whose legs you restructured. `HARNESS --list-legs` prints each gate (`LEG  NAME`)
  without running a leg body. Every gate also checks its leg's NAME, in every run: a name selects
  one leg; a leading leg number (`80:`, `2-3:`, `59-63, 66:`) numbers one leg, a further leg beside
  it taking a suffix letter (`80b:`, `40-47b:`); and where the harness header declares its groups
  as `#NAME (LEGS)` (the members of `tools/test-test-run.sh`), each leg closes with a declared group that lists
  its numbers, and each listed number is carried by a leg of that group. A naming defect is an
  ordinary `FAIL  leg naming: ...`. Gated and alone-clean so far: the six members of
  `tools/test-test-run.sh`,
  `tools/test-mutation-run.sh`, `tools/test-promote.sh`, `tools/test-machine-verify.sh`,
  `scripts/test-setup-ocaml-env.sh`, `tools/test-merge-build-check.sh`,
  `tools/test-merge-build-replay.sh`, `tools/test-row-trace-compact.sh`,
  `tools/test-affected-tests.sh` and the two `benchmarks/test-*box*.sh`.
- **A leg that cannot go on ends itself, never the harness** (gh-ocannl-1374). Each leg of
  `tools/test-test-run.sh` is a function run by `harness_leg_run 'NAME' leg_N`; a check it cannot
  get past (an extraction or mutant that matched nothing, a fixture repository that will not
  commit, a fixture that never became ready) ends it with `harness_abandon DETAIL; return`: one
  `FAIL  NAME: abandoned, its later checks did not run`, and the legs after it run and report.
  An `exit` there ended the run and hid every later leg, the shape one leg's code drift kept
  costing; the body's own last status is no verdict, only `report`/`skip`/`harness_abandon` are.
  A shared fixture outside the legs that is not what they assume is a FAIL where it is built and
  the run goes on; only a subject that cannot be staged at all (a missing script, the
  sourced-files scan `stage_shared` runs) still exits 2. An `exit` that does escape a body is
  named: `aborted before finish (rc=N) inside leg: NAME`. Leg 86 pins it on a copy of the
  harness (leg 79 abandoned, leg 70 after it still reporting) against that copy with the check
  spelled as an `exit`; `scripts/test-harness-support.sh` pins the primitives. `set -u`'s
  unbound-variable abort is not containable this way and still ends the run.
- **A run the host refuses as a whole is abandoned once, never failed per check**
  (gh-ocannl-1940). Another dune holding `_build`'s lock refuses every build a harness's subject
  asks for (`Error: Another Dune instance is currently running. Aborting...` on dune 3.24), and
  `tools/test-typecheck-scan-cases.sh` used to read that as 53 FAILs, with 18 PASSes from the
  checks that expect the subject to fail: a worker's own concurrent run, read as a broken tree.
  `tools/typecheck-scan-cases.sh` now exits 75 with `LOCK HELD` for it (from its own engine build,
  or from the dune its engine runs for an owning module's context), and the harness ends there
  through `harness_abandon_run REASON`: one `ABANDONED  REASON` line, a footer counting what was
  decided before it, exit 75, no `FAIL`. It is the exception to the rule above, for a refusal no
  later leg escapes and no defect of the tree explains; the remedy is to run it again once the
  other dune is done (one dune per worktree at a time). The subject's exit status is the signal,
  never dune's wording read by the harness; its leg 13 holds the lock of a scratch build
  directory with a real dune, so a dune that rewords the refusal turns that leg red.
- A negative status needs the refusal's reason as well (gh-ocannl-2071). Exit 2 from
  `tools/typecheck-scan-cases.sh` covers both an engine build failure and invalid fixtures;
  its harness's `expect_rc` therefore requires a matching stderr reason for every exit-2
  expectation. Leg 14 forces the engine build to fail and requires both leg-5 checks to FAIL,
  with no PASS; the former status-only helper passes both. The other `expect_rc` harness,
  `tools/test-ppx-output-diff.sh`, requires a reason in the helper too, including usage
  refusals, build failures, empty scopes and unresolved revisions. The inline mutant controls
  in `tools/test-affected-tests.sh` pair their status with the expected UNMAPPED path or
  failed alias-listing diagnostic; the spelling of a helper is no boundary for this rule.
- **A harness whose checks are bare commands under errexit runs each leg contained**
  (gh-ocannl-1539): `harness_leg_run_errexit 'NAME' leg_N` runs the body in a subshell of its own
  (`harness_errexit_subshell`, a function whose body is a `( )`, so its `set -eE` is its own and
  the errexit scan reads the file precisely), so a failed check, an `exit` or a `set -u` abort ends
  that leg alone, abandoned with the failed command and the function it ran in, and a body passes
  only by reaching its end. The subshell exits 0 through its own EXIT trap and hands its verdict
  back in marker files: wrapping it in `|| rc=$?` would not do, since bash ignores errexit in the
  whole body under a condition, even a `set -e` inside it -- a call there is caught by a probe and
  abandoned as never run. Being a subshell, a body's exports and `cd` stay its own, and so would
  its counts: `report 1`, `skip` and `harness_abandon` inside it end it, abandoned with their
  text. What it changed on disk outlives it: the harness defines `harness_leg_restore`, run
  contained after every abandoned leg (`tools/test-mutation-run.sh`'s puts back the stand-in
  `cmp`, a moved lock, a live run; its leg 22 pins it on copies of the harness, uncontained and
  unrestored controls included). Under bash 3.2 a `set -u` abort reaches the EXIT trap as status
  0, which is why completion, not status, is the verdict.
- Strict skips: the toolchain-installing jobs set `OCANNL_TOOL_HARNESS_STRICT=1` (every skip is a
  FAIL); the toolchain-free `harnesses` job runs `tools/test-harnesses.sh --shell --expected-skips
  tools/harness-skips-ci-shell.txt`, strict except for the `HARNESS<TAB>LABEL` lines listed
  there, and red on a listed skip that no longer occurs (gh-ocannl-1236). A leg that newly skips
  on Ubuntu is a defect to fix or a line to add, with its reason in a comment above it; the
  list's line forms are a fail-closed allowlist (`tools/test-harnesses.sh`'s header).
  The job's command is `tools/test-harnesses.sh --ci-shell`, and it reproduces that check on any
  box with a toolchain (gh-ocannl-1593; under `execution slot` on a fleet box): it hides dune from
  every leg that probes a real one through `harness_real_dune` (`OCANNL_TOOL_HARNESS_NO_DUNE=1`),
  and the fleet tables `benchmarks/test-check-box.sh` reads, so a Linux box shows the job's
  skip set and macOS adds the skips `tools/harness-skips-ci-shell-darwin.txt` lists (none now). A
  leg that needs a real dune takes it from `harness_real_dune`, never a probe of its own: one
  `--ci-shell` cannot hide runs green here and skips, unlisted, only on the job.
- A leg whose subject does not exist on this platform declares itself not applicable rather than
  skipping (gh-ocannl-1839): `harness_not_applicable REASON; return` as the first decision of a
  `harness_leg_run` or `harness_leg_run_errexit` body prints `N/A   LEG` with the reason, is counted
  apart in the footer (`all legs passed (0 skipped, 1 leg(s) not applicable)`), and passes a strict
  run with no list naming it, since nothing was lost; a skip is coverage the host lost, so the
  lists stay strict and per-platform skip lists are not the tool for it. Decide it from a probe
  that OBSERVED the subject absent, never one that could not tell: leg 1b of
  `tools/test-refusal-manifest-sync.sh` is N/A only where the host's `/bin/bash` expanded an empty
  array under `set -u` (Bash 4.4 on, CI's Ubuntu), and its leg 1c is the probe's negative control.
  The Linux-only process facts are declared the same way on macOS, so the Darwin list holds no
  line for them (gh-ocannl-1930): `tools/test-test-run.sh` leg 67b where `/proc` holds neither
  `locks` nor `fdinfo`, its leg 85 (the child subreaper) where `uname` named another kernel than
  Linux (67c and 85b are the probes' controls, over stand-ins), and
  `scripts/test-setup-ocaml-env.sh` leg 1b where `killpg` was seen to answer dead for the leg's
  own zombie-only group. The outcome is a leg's, so a check that can be not applicable is a leg of
  its own (a `67b` beside `67`), never a label inside a leg that otherwise runs; and a facility a
  LINUX host lacks (no python3, a refused prctl) stays a skip.
  A declaration outside a function leg body or in a subshell of one, with no reason, after a
  check, twice or followed by one is a FAIL, never counted. `scripts/test-harness-support.sh` pins the primitive, and
  `tools/test-test-harnesses.sh` that `--ci-shell` still refuses an undeclared skip beside one.
- The real-dune checks run per PR on a job of their own (gh-ocannl-1650), not by giving the shell
  job a dune: its no-dune skip set stays a test of its own. Their skips under `--ci-shell` live in
  `tools/harness-skips-ci-shell-dune.txt`, which `--ci-shell` joins to the job's list and holds to
  exactly the skips the hidden dune caused, told by the reason `harness_real_dune` gives
  (`harness_dune_hidden_reason`; `skip()` records them in `OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD`):
  such a skip listed elsewhere, or a dune-list skip with another reason, fails the tier. The
  `harness-real-dune` job installs dune and runs `tools/test-harnesses.sh --ci-real-dune`: only
  the legs `real_dune_legs` selects (an `--only` ERE per member), strict, and red unless every
  dune-list check printed `PASS`. A new real-dune check therefore lands as a dune-list line, and
  its leg's number in the selector if it is a new leg -- the job names the check it did not see
  pass. A nested run of a harness copy unsets the record variable with the others.

- Shell harness checks capture failures with `command || rc=$?`, preserving the caller's
  `errexit` state; an unconditional `set -e` after a check makes later unguarded failures
  terminate a harness that started with only `set -u` (gh-ocannl-1178).
  Shared `harness_cleanup` prints `aborted before finish (rc=N)` if `finish` never ran,
  including signal exits, and still cleans scratch or identifies it with `--keep`.

- `tools/test-run.sh` is the one way to run `dune runtest` / `dune build @slow` from a session;
  its header documents usage. It exists because every hand-rolled alternative has failed in
  practice, each differently: piping dune to `tail` reports tail's status (no pipefail), so
  promotion diffs read as green; a wrapper variable named `status` is read-only in zsh, so the
  assignment dies before the sentinel prints and a green suite looks failed; waiter loops on
  `pgrep -x dune` match the editor's immortal `dune ocaml-merlin` daemons and spin forever (one
  review session accumulated ten stranded waiter shells); `kill -0 $pid` can latch onto a
  recycled pid; and an uncapped hung run (macOS XProtect stalling a fresh exe, a wedged backend)
  strands whatever waits on it. The script runs dune unpiped under a wall-clock cap that kills
  the whole process group, records the verdict in a FILE (`wait` polls that file under a hard
  timeout — it structurally cannot strand), and holds a per-worktree flock so a second
  invocation refuses loudly instead of queueing behind dune's lock — "I lost track of a run so
  I started another" being the usual start of the spiral. A run is two processes — the launching
  shell, which takes the lock and publishes the run, and a perl supervisor that inherits the lock,
  caps its child and records the verdict (gh-ocannl-606 collapsed the former three-party launch and
  its publication handshake). The child is the script's `_resolve` first — the batch's backends,
  width and fleet slot — which then runs dune as its child and, once dune exits, records dune's
  promotion list before exiting with dune's status: work that must run under the lock before or
  after dune belongs in those phases, never in the launcher, which then needs its own copy of
  the cap, the signal relays and the group kills (staging#832 grew one, and six of its nine review
  rounds went there; gh-ocannl-1106 deleted it). Everything a run keeps per worktree (the lock, its owner pointer, the
  `last` pointer) lives under `~/.ocannl-test-runs` keyed by the worktree's path, never in the
  tree: a run leaves the worktree exactly as clean as it found it, so a teardown that judges a
  worktree by its ignored files is not refused over lock residue. `OCANNL_TOOL_TEST_RUNS` relocates
  the lock with the diagnostics, so two sessions on one worktree see each other's lock only under
  the same override. The read-only `paths FIELD [RUN|last]` contract (gh-ocannl-671) returns one
  absolute path: `run` defaults to the last recorded run; `worktree`, `runs`, `lock`, `owner`, and
  `last` default to this script's worktree, or use an explicit run's recorded metadata. `last`
  names the pointer file. An absolute run reference does not require a valid current state
  root or `HOME`; relative current-state queries ignore `CDPATH`. `lock-status [RUN|last]` without a run probes this worktree's current
  and legacy locks, with a run only its recorded lock; it prints `idle` (0) or `held` (3), with
  inspection/argument errors 2. A present invalid `runs` entry is an error; only its absence
  identifies legacy metadata. Recorded path entries must be exactly one LF-terminated absolute
  path valid on this host (drive forms are Windows-only). Query paths are line-oriented: CR/LF paths fail with exit 2;
  this does not redesign the existing launch metadata format. Queries create no state, even before the first run, and are
  snapshots, never reservations. Missing directories are supported; parent traversal
  must resolve through the launch shell or fail with exit 2. Native Windows drive spellings
  can resolve `missing/..` where POSIX spellings cannot: queries follow the shell's identity.
  A missing suffix whose symlink-parent prefix differs between logical cd and physical
  traversal is refused; already resolved roots and ordinary symlink prefixes work.
  The harness uses them for fixture pointers, run discovery, and repeat lock assertions;
  deliberately tested process predicates/identity fixtures still extract functions. Prefer foreground `run` launched through the agent harness's background mode
  (the harness notifies on exit); `start`/`status`/
  `wait`/`stop` are only for runs that must outlive the launching session. Its own options go
  BETWEEN the subcommand and the dune arguments (`run --cap 900 build @alias`); both
  misplacements are refused with exit 2 having run nothing — the option written AFTER the target
  by a guard that knows the correct order and names it, before dune is spawned. Every OTHER
  invocation dune's own CLI refuses — an unknown option, an unknown subcommand, a missing or
  malformed operand (`dune build -j x`) — is dune's to refuse: it prints `dune: <complaint>` then
  `Usage: dune …`, exits 1 and runs nothing, and the digest recognises that pair at the head of
  the log (whole, first and alone — a `dune:` line alone could be a test's own output, and
  anything after the `Usage:` line beyond dune's `Try '… --help'` and the `exit:` sentinel is
  evidence that something ran, such as a `dune exec` program printing a nested refusal) and reports
  `INVOCATION REFUSED (dune rejected the arguments; nothing ran)` quoting dune's complaint,
  instead of the `FAIL (exit 1)` plus `no Error/File lines matched` that once sent a session
  debugging code that was never compiled (staging#652, gh-ocannl-944). Past dune's own `--` the
  same word belongs to the executable and is passed through. The verdict vocabulary, with the
  exit-code contract a caller branches on without parsing the log: `pass` (0); `FAIL` (dune's own
  status, with a fingerprint of the `File "…"`/`Error` lines); `INVOCATION REFUSED` (`run` and
  `wait` exit 2 — the script's usage code, shared with its own refusals, while the RECORDED status
  — the log's `exit:` sentinel, `status`, `list` — stays dune's 1: it is what happened); `BUILD
  FAILED` (65 over a recorded 1, gh-ocannl-1547: the run's own trace holds a process that exited
  nonzero producing only compile artifacts — a type error, a ppx or link failure — so the tests
  needing it never ran; the digest names the step, and a named test's `.exe.output`/`.actual` no
  process of this run wrote as NOT this run's output, which is how a mutant that did not compile
  once passed for a caught one and an earlier run's output for the mutant's); `TIMEOUT`
  (142, the cap's SIGALRM exit and the only code allowed to say timeout); `KILLED` (137);
  `CANCELLED` (129/130/143); `ERROR (toolchain/setup: nothing ran)` (126/127). `wait`'s own
  deadline is 124, and `status` exits 0 published / 3 running / 1 died without a verdict — it
  reports publication, never the verdict. `tools/test-test-run.sh` legs 29–32 pin all of this
  against a fixture dune that emits dune's exact refusal stderr, with the red-run and guard controls.
  One refusal is asked of dune BEFORE the batch (gh-ocannl-1448): a `dune build` alias operand dune
  does not know (`@test/operations/runtest-misspelt`) used to surface only after the launch had
  resolved the backends, waited for and taken the fleet slot and probed the device. `_resolve`
  now first asks `dune show aliases DIR...` (a listing that builds nothing; dune 3.24 prints it on
  stderr and exits 0 even for an unknown directory, answering a query that names one with
  nothing but the error), and an alias absent from its directory -- and, under `@`, from every
  dune directory below it -- is `ALIAS REFUSED` (exit 2, nothing ran; `plan` says run would refuse
  it). It refuses only on an answer that reads as dune's (every known directory lists the standard
  `default`), and only for an argv whose options are all on its allow-list of ones that cannot
  change what aliases dune defines (`--profile`, `--workspace`, `-p` and the like leave the batch
  to dune, as does a context-qualified `@_build/CONTEXT/...`). The directories below an operand's
  are those holding a `dune` (or `dune-file`) and those its `(subdir ...)` stanzas give rules to,
  read by a deliberately small s-expression reader that is fail-closed on what it does not model
  (`#|`, `#;`). The census runs before any query, and a tree it cannot list completely -- a file
  it cannot read, or one importing another file anywhere (`include`, which can import a subdir or
  a `dynamic_include`; a `dynamic_include`, which loading that directory's rules would build
  outside the slot), decided on its tokens -- leaves the whole batch to dune. So a fixture dune with no `show` branch leaves the aliases to dune; a fixture that
  records its calls must answer `show` without recording it. Leg 89 pins it against a table-driven
  fixture listing and, where a dune is at hand, a real one.
  `repeat [--alone] N` is the supported flake diagnostic (`--direct` for per-test aliases: one
  build, then replays; see Dune mechanics below): it holds that same worktree lock once,
  gives every Dune invocation a freshly cleaned cache-disabled build context, retains each
  iteration's separate stdout, stderr and exit status, and writes every pairwise diff. Stdout or status drift is red;
  stderr-only drift is called out separately and stays diagnostic-green. `--alone` adds `-j 1`,
  making the no-sibling rerun that distinguishes resource contention from intrinsic instability.
  The set is published as this worktree's `last` run, `wait last` scales its default deadline by
  the iteration count, and `stop last` signals its outer coordinator so cancellation ends the set
  rather than merely killing one iteration and starting the next. Cancellation remains deferred
  through comparison, and an exit finalizer publishes the verdict with further group signals
  ignored, so even a finalizer child killed by the original signal cannot strand the managed run.
  Cancellation state and that finalizer are armed before the run is published as `last`; each
  iteration rechecks cancellation immediately before and after supervisor launch.
  If a supervisor is killed while its identity-verified Dune process group survives, the group is
  reaped before another iteration can reuse—or final cleanup can remove—the shared build tree. Each
  launch also inherits a FIFO writer: EOF after the supervisor exits proves a child did not escape
  the recorded group with `setsid`; an unclosed witness retains/refuses the build tree. A
  prior nonzero iteration remains the set's exit status when a later iteration is cancelled. The
  dead supervisor pid is cleared before the potentially slower process-group reap. That reap gates
  on group reachability, not the fallible process census, and revalidates the recorded leader token
  before escalating from TERM to KILL so a recycled numeric group is never targeted. Identity
  matching deliberately admits the original zombie leader; after KILL, a verified zombie-only
  residue is inert and does not replace the iteration's timeout/cancellation verdict with an error.
  A reachable group that loses leader identity before KILL is refused without consulting the
  fallible census. After KILL reached a verified group, leaderless residue is accepted only when
  its census is zombie-only; any live leaderless descendant is a loud refusal to reuse the tree.
  Those zombie-only shapes reproduce only under a parent that never reaps orphans (staging#1172's
  three review rounds each found one in the reviewer's container; macOS and the Ubuntu runners
  reap). Leg 85 of `tools/test-test-run.sh` (Linux, gh-ocannl-1376) runs `repeat` and `run` under
  a stand-in: a python3 child subreaper (`prctl(PR_SET_CHILD_SUBREAPER)`) that waits for its own
  child only, so every orphan below it stays a zombie in the group it left, and that records what
  it held. A capped set keeps TIMEOUT, a GPU set passes through the resolution's inert branch
  (`left only zombies ... inert, not reaped`), `run` keeps its verdict, and the stand-in must have
  held a zombie and nothing live; taking out the iteration shell's reaping trap, or that branch,
  makes the set refused (exit 2) under the same stand-in.
- `last` is the worktree's newest PUBLISHED run, and a launch publishes only once the script runs
  it (after the worktree lock, before the fleet slot it takes itself). A launch queued behind a
  wrapper that waits first — `fleet-worker.sh execution slot --wait-window -- tools/test-run.sh
  run ...` waiting out a measurement or a free slot — is no run yet, and `wait last` used to
  answer with the previous run's verdict: a false green, three times in wave-20261102
  (gh-ocannl-1830). So `status last` and `wait last` follow the worktree: while a launch of it is in
  flight unpublished, a finished `last` is not their answer (`queued:`, status 3; `wait` waits
  for the launch to publish and then for its run, 124 when the budget runs out, so the bounded
  re-issue loop works unchanged), and once one they saw ends unpublished (its slot refused) they
  refuse to answer for the run before it (`unpublished:`, status 1, wait 2) until a run publishes;
  a named run (`wait RUN`) is waited for alone. Nothing on disk records a queued launch, so it is
  read off the process table, from each process's EXEC ARGV and never from shell text: a
  fail-closed allowlist (`proc_table_perl`: a shell, env, perl or python program; at a command
  position -- the first word, or the first after a `--`, the separator of every fleet wrapper,
  with env/nohup/timeout/interpreter prefixes peeled -- a `test-run.sh` word naming THIS
  worktree's script, followed by a launching verb; no `ssh` or `execution leg` before it; not the
  reader's ancestors, nor a fork carrying another live process's argv, such as the slot wrapper's
  double-forked guardian), less the chains this worktree's published runs recorded at launch
  (`launchers`: each launcher and its ancestors, by pid, start token and an MD5 of the argv), so a
  run's own wrapper living on after its verdict is not mistaken for a launch to come. A `-c`
  string is never parsed (staging#1678 spent rounds 2-7 on a shell-text reader, one corner per
  round, before it was cut back to argv): a launch a shell's own control flow still holds is
  seen once its own process runs it (leg 105 pins it: `last` is A while B is still the shell's
  text, and B is followed once its process runs). A relative script word is resolved where the
  command runs: an `env -C DIR` or `--chdir=DIR` prefix carries that directory through the peel
  (gh-ocannl-1913; before, a wrapper outside the worktree carrying `-- env -C /repo tools/test-run.sh
  run` was missed and one in it moving to a sibling over-read). A launch of the worktree under
  another OCANNL_TOOL_TEST_RUNS is no launch under this root when its argv says so (an `env`
  assignment, or `-u` or `-i` before the script word: the default root; a relative root is the
  worktree's, where the script cds first) — and only so: process environments are never read, since
  /proc/PID/environ is the exec-time one and a wrapper that exports a root before it runs its
  command shows the old one there (staging#1788 round 4), so a root exported to a launch is the
  loud over-read: a wait that times out, then `unpublished:`. A default root (`-u`, `-i`) is
  under the HOME the prefix names, else unknown, and a launch under an unknown root is here; a
  root is read literally (`~` unexpanded), as the script reads its own, and under both the logical
  and the physical reading of a symlinked `..` (bash's cd takes either, by what exists): a match
  under either keeps the launch. The
  one fact stored is the sighting (a `queued-after` mark in the run `last` named then, or
  `queued-<key>` before the first run), since a launch that left without publishing leaves
  nothing else; a mark is a file's existence per run, so no late reader can overwrite a newer
  sighting. This
  does not settle gh-ocannl-607: the gap is before any publication, so deriving `last` from the
  run directories would leave it exactly as it was. `list` lists this worktree's runs (by `wt`,
  `--all` for the state root's): the newest entry of a shared root is as likely a sibling
  checkout's, and an `ls -td` there read one. Legs 99-100 and 105 of `tools/test-test-run-last.sh`.
- Every liveness question in that script — per pid and per process GROUP — reads process STATE and
  not only the signal, because `kill -0` succeeds on a ZOMBIE exactly as on a live process, and an
  identity token does not rescue the check either: a zombie leader still prints its recorded
  `lstart`. Answered with the signal alone, `stop` could announce `orphaned process group N ignored
  TERM; escalated to KILL` for a group holding nothing but corpses — the one report someone consults
  when working out why a worktree lock will not clear (gh-ocannl-742). `group_alive` is ONE shared
  definition, `scripts/process-group.sh`, sourced by both `tools/test-run.sh` and
  `scripts/setup-ocaml-env.sh` for the identical misreading (staging#621 replaced the two
  synchronized copies), with `pid_alive` beside it for one process (gh-ocannl-1383: a cancelled
  sweep unit, orphaned in a group of its own that the sweep's group check never sees, read as a
  survivor while it waited for its reaper; `sweep_harness` asks every "outlived" question through
  it, and a failure prints the survivor's `ps` row, where `Z` names the misreading): `/proc` where
  there is one (fork-free, through the shell's `read`), `ps -A -o pgid=,stat=` on the BSDs and
  macOS, degrading to the bare signal only where neither answers; the signal probe stays FIRST as a
  necessary condition, which makes the predicate a strict narrowing of the `kill -0` it replaced —
  it can turn a phantom alive into dead and never the reverse. A state read is still a CENSUS, and a
  census is a SNAPSHOT — a child forked while the glob is being read is not in it, and a leader that
  exited into a zombie during it is — so both callers let that answer shorten a reap or reword a
  report but never SKIP or DOWNGRADE one: TERM and KILL both go out on reachability alone, and
  liveness decides only whether the grace is worth sitting out (asked AFTER the TERM, so a member
  the earlier census could have missed is included — a wait has a point only where something can
  still act on the signal) and which sentence the operator reads. A census allowed to veto cleanup
  buys the phantom back as a survivor mutating `_build` behind a released worktree lock; one
  allowed to skip just the TERM costs a child the chance to flush its output and release what it
  holds. Both are worse than the phantom. Whether the bare probe over-reports at all
  is a property of the kernel, so a local pass proves less than it looks: Linux (and every container
  on it) counts the zombie and says alive, while Darwin's `killpg` already answers `ESRCH` once a
  group holds only corpses, and under a PID 1 that does not reap the zombie is PERMANENT, so a retry
  loop around the signal was never the fix. That is why the two harnesses run where the fact
  reproduces: `tools/test-test-run.sh` and its sibling `scripts/test-setup-ocaml-env.sh` are on no
  dune alias (they spawn, STOP and kill processes, and the `bounded` legs sit out watchdog
  timeouts) but since staging#621 (gh-ocannl-795) CI runs BOTH in one isolated step on Ubuntu with
  no OCaml toolchain — now the `harnesses` job ("Shell harnesses"), beside the build matrix rather
  than ahead of one of its shards — summing their exits so a red first harness does not suppress
  the second's diagnostics. The step is now the home of any hand-run harness that has a reason not
  to be a dune test — `tools/test-ci-times.sh` and `tools/test-ci-shard.sh` joined it for a
  different one (their subjects invoke `python3` under that exact name) — and they share one
  contract for a leg the host cannot decide:
  `SKIP LABEL REASON` on stdout, a footer that always prints the skip count
  (`all legs passed (N skipped)`), and exit 0 when no leg FAILED — so "all legs passed" over a run
  that decided fewer legs is never the reading, and a macOS run of the sibling is green with its
  zombie-only bare-signal comparison skipped rather than hard-failed as vacuous, which is what
  lets one step run them all on any runner. `tools/test-test-run.sh` sources
  `group_alive` from the working-tree helper, pins that exactly one production definition exists
  across the helper and its two callers (with a synthetic duplicate as the negative control), and
  builds a group holding nothing but a zombie, asserting both the claim and, by shadowing `kill` so
  the signal probe is forced to answer alive, the portable control that the state reader alone
  rejects that group: on Ubuntu the real group shows bare `kill -0` alive and the reader rejects
  it, on Darwin the forced-signal control is the only zombie evidence, so read the skip count
  before treating a local green as coverage. It reads states and groups
  through its OWN `/proc`-then-`ps` readers, probed against a known-live process before use: a Git
  Bash/MSYS `ps` takes no `-o`, and a leg that cannot tell "not a zombie yet" from "gone" must SKIP
  rather than pass or fail — an unreadable state made both zombie assertions fail there and let the
  cleanup assertion pass vacuously.
  The same harness drives `stop` itself, keeping its two surviving-group sentences apart:
  a FORGED run directory (`cmd`, `cap`, `wt`, `log`, `pgid`, `gtoken`, and deliberately no
  `pid`/`exit`, so the surviving-group branch is the one reached), pointed at by `last` under
  a private `OCANNL_TOOL_TEST_RUNS` so the ambient run history is never touched, with the pointer's
  filename returned by `paths last`, rather than reconstructing a worktree key. Against it: a group whose
  leader ignores TERM — reported as ignoring it, and the escalation separately checked to have
  actually killed the WHOLE group, since announcing a KILL it did not send would leave the worktree
  lock held — and one whose leader takes the TERM, reported as TERMed with a re-run asked for. The
  two legs differ by one `trap` and nothing else, so a `stop` that worded them alike fails exactly
  one, and each is matched against stop's whole output rather than a substring of it: both wording
  legs pass a containment test against a stop that prints both sentences at once.
  Every fixture group holds TWO processes for the same reason — with only its leader in it, the
  incorrect leader-only `kill -KILL "$pg"` passes for a group kill while real dune children would
  survive it. And the leader is recorded for cleanup BETWEEN the fork and the checks, not after
  them: job control has just put that child out of reach of a signal aimed at the harness, so an
  interrupt in that window otherwise leaves the group running past the run (both: Codex round
  1 on staging#505, each reproduced against a mutated copy before being fixed). The sentence is
  matched against stop's STDOUT alone, with stderr kept and shown only on failure: a diagnostic on
  stderr is not part of the answer, and merging the two made a passing `stop` fail an exact match.
  The fixture's second member — a GRANDchild, so nothing holds it as a zombie — is killed only
  under its recorded start token, the same gate the shipping script applies to any pid it did not
  itself fork, since `stop` having already killed it frees its pid for recycling before the EXIT
  trap runs.
- `tools/test-test-run.sh` covers the `run`/`start` lifecycle (gh-ocannl-967): readiness precedes
  status/stop/signals; completion checks verdict, sentinel, group and lock; competing launches
  create no run directory. POSIX legs cover INT/HUP/group TERM, orphan recovery and background
  children that close their locks. The forged publication window records settle sleeps before
  checking reaping; `last` squatter/legacy lock cases end with a byte comparison of the fixture
  tree. Fixtures use `paths`/`lock-status`, identity-checked cleanup and bounded child waits.
- `supervisor_perl` blocks INT/TERM/HUP/ALRM across its fork expression (gh-ocannl-1141): Perl
  can deliver a deferred signal before assigning the returned pid. The parent restores the original
  mask after assignment; the child restores it after resetting all four handlers to DEFAULT.
  `tools/test-test-run.sh` leg 74 injects TERM into both branches, with unmasked orphan-lock and
  inherited-handler duplicate-verdict controls. Reader cancellation legs wait for `FAKE_REACH_CALLS`.
- **A group signal reaches the probe as well as the prober** (gh-ocannl-1362). A shell that traps
  INT/TERM/HUP survives a signal sent to its process group, but the `cat`/`ps` children of a
  liveness probe it is running at that moment do not, and a probe whose child died answers "gone".
  The attached `run` launcher read that as a supervisor that died without a verdict, while the
  supervisor, TERMed by the same group signal, was reaping and about to publish (the CI flake: exit
  1, lock held, verdict file appearing moments later). Bash runs a trap before the interrupted
  command's caller sees its status, so `sup_alive_settled` counts trapped signals and re-asks any
  probe one landed in. The lifecycle leg reproduces it deterministically with a `cat` shim that
  sends the group TERM from inside the wait's first probe, against a plain-`sup_alive` control;
  the unshimmed GROUP_TERM leg hit it only under load (one run in 25 at background QoS on a busy
  mac-studio, none in 35 at normal priority).
- **A trapped signal can be held in front of `wait PID` until PID exits** (gh-ocannl-1504). Bash
  checks for pending traps between commands; a signal landing after the last check and before
  `wait` blocks has its trap run only once PID exits. A shell relaying its signals to a child it
  waits on (the attached `run` launcher, the `repeat` coordinator) then relays the cancellation
  after the run it was meant to stop is over. Isolated (a trapped-TERM relay looping a probe and a
  `wait`), bash 3.2 held 2 TERMs in 4000, bash 5.2 1 in 20000 under CPU load; in
  `tools/test-mutation-run.sh` leg 10 it was the 45-second timeout about once in 1400 runs on the
  hosted runner (once in 70 on mac-studio beside `tools/test-harnesses.sh`), which a kernel signal
  trace pinned to the launcher: TERM delivered to its handler, nothing relayed for 45 seconds. Those waits go through `wait_tick`, a
  one-second foreground Perl tick that ends early when the child does, so a held trap runs
  within a second. Every PID-directed cancellation now waits for the current tick, up to 1 s;
  a group signal ends the tick immediately. A foreground tick is reaped before an exiting trap
  runs; the previous
  background tick could outlive a cancelled sweep, caught by its group-liveness oracle.
  `scripts/process-group.sh` owns the shared `wait_tick`; callers that
  need the child's exit status use `wait_child`, with `$!` of a simple background command
  (not a pipeline's last process). It polls this shell's running and stopped
  jobs before the final, nonblocking `wait` (`jobs -p` alone can retain completed jobs on
  Bash 3.2, caught by the sweep integration). A recycled numeric pid cannot extend that
  wait. Its job-table substitution ignores INT/TERM/HUP so a group signal cannot empty the
  query and turn the final wait into a blocking one. Besides `test-run.sh`, the relays in
  `sweep.sh` (unit and lane), `bisect-steps.sh`, `ci-compiler-test.sh`, `harness-legs-alone.sh`
  and `benchmarks/gh834_cells.sh` use it (gh-ocannl-1914). The audit covered the whole repository,
  including hidden paths and embedded shell: the benchmark drivers `gh612_cells.sh` and
  `gh1133_cells.sh` wait after kills; other source hits were fixture cleanup, comments,
  documentation or Python process waits rather than shell relays.
  `machine-verify-far.sh`'s watcher wait follows its kill and has no signal relay;
  `harness-support.sh`'s child wait follows a bounded liveness check/kill, and its traps exit.
  `test-harnesses.sh` probes numeric PIDs with `kill -0` before ordinary waits; its cancellation
  waits follow the relay. Cleanup waits after delivery of the cancellation are not the wait-entry relay race.
  `scripts/test-process-group.sh` drives 20,000 TERM deliveries to `/bin/bash` under fork/exit
  load with the shared wait. A shadowed `wait` deterministically rejects a running or stopped
  job; a bare-wait twin loses that structural check. Status cases pin 37/41/43/143 and non-child
  `127`. An unbounded foreground wait is the deterministic control of the response oracle.
  The shared verdict has its own 600 s budget and runs before the bare-wait diagnostic. That
  diagnostic searches up to 4,000 deliveries under a separate 60 s budget, stopping each lane
  at its first held signal; budget expiry stops only the diagnostic. No race observed is
  diagnostic, not a CI failure. Bash 3.2 crashes in that control are counted apart and never satisfy the held-signal
  observation; a crash in the shared-wait leg fails. The harness tracks and relays TERM to its
  Python driver through the same shared wait; a foreground-launch twin loses the cancellation
  deadline while every trial group is reaped.
  Each child acknowledges startup before the relay publishes readiness;
  the 2.5 s response bound starts there, not at spawn. A signal relayed this way needs no
  deadline sized for load; a leg timing out on one is a lost relay first.
- **A short bound that kills a stand-in cannot also be the time it gets to start** (gh-ocannl-1366).
  The device-probe hang legs (`tools/test-test-run.sh` 76 and 83) ran their never-answering
  stand-in under a 1 s bound and then read the call and pid it records first thing; on a loaded
  mac-studio its start alone outlasted the second, the bound KILLed it before it wrote a byte, and
  the leg failed with an empty record (`the query was not asked once`) while the verdict under test
  was right. The clock starts at the fork, before `exec`, so no stand-in is fast enough to rule
  this out. `probe_hang_run` now judges only a run the stand-in started in: one that left no start
  record is repeated under a bound three times as wide (up to 27 s), with a `note:` line; a
  stand-in that answers runs under a 30 s bound it never waits out. Leg 76 pins the
  widening with a stand-in that records nothing on its first run (a marker file, not a delay: the
  bound is polled, so a delay only races it), which must be judged in the second, at 3 s. A
  stand-in delayed 1.5 s before recording reproduced the original failure on the base tree; a
  natural reproduction was not found (mac-studio at load 70, CPU and fresh-exe churn, started it
  within 0.2 s).
  The same trap reddened `sweep_harness`'s probe legs and the benchmark drivers' capped cells on a
  loaded macbook-pro (gh-ocannl-1807), fixed by making the start-up irrelevant rather than
  waiting it out. `sweep_harness` counts a unit's queries from the SWEEP's record of them
  (`probe_queries`), which a query killed before its first line is still in; only the answered leg
  reads the stand-in's own record (its directory), and the hung query's pid is judged when the
  stand-in got to publish it. `SWEEP_TEST_PROBE_SILENT` pins the count against a stand-in that
  records nothing. In `benchmarks/`, a capped fixture writes its evidence and then `publish_pid`s:
  `cap_counted_from(pidfile)` (`benchmarks/test/test_cell_group.py`; `started=` on
  `CellTimeoutTest.run_cell`) holds the wait that carries the cap until the publication, and
  `interrupt_once_published` replaces an interval timer armed before the spawn. A new capped or
  interrupted fixture whose test reads what it wrote before the kill goes through one of them.
  `test_every_capped_evidence_fixture_waits_for_publication` guards that structure with synthetic
  controls, on `@benchmarks/runtest-capped_fixture_scan` and `@test/operations/scans`. Its AST
  dialect covers literal short timeouts, patched timeout constants, alarms, subsequent assertions
  or file reads, and same-class forwarding wrappers; its docstring states the boundary, including
  computed caps and control-flow ordering it does not prove.
- **`/proc/locks` names a flock's ACQUIRER, not its holders** (gh-ocannl-1107): the lock lives on
  the open file description, so after `take_lock`'s perl exits the line keeps its dead pid (0 from
  inside a pid namespace; some kernels drop the row) while the supervisor, dune and a `setsid`
  leftover hold it through fd 9.
  Trusting the name, `stop` reported `STILL hold` and reaped nothing. The census `stop` and the
  supervisor share, `lock_census_perl` in `tools/test-run.sh` (gh-ocannl-1906), never reads
  `/proc/locks`: on Linux it sweeps every `/proc/<pid>/fdinfo` for the FLOCK on the lock's device
  and inode, elsewhere it takes `lsof`'s openers, bounded (2 s in the supervisor, 10 s in `stop`).
  Each pid is paired with the start token it carried from before the census saw it until after,
  and signalled only while it still carries that token.
  Leg 67 pins it with a lock whose acquirer exited, on Linux and macOS. Its negative control
  (Linux) is a census that goes by the pids `/proc/locks` names.
- **A start token has one Perl renderer, and every rendering is pinned to it** (gh-ocannl-1944).
  Tokens are compared as strings across writers — `ptoken`/`gtoken` from the supervisor's Perl,
  the shell's re-checks through `ps_token`, a launch chain's from `proc_table_perl`'s bulk table —
  so one rendering that differs (a missed trim, an unpinned time zone) makes a live run read as
  dead. `census_token` is the one Perl renderer of a pid's token (the supervisor calls it on
  `$$`); the bulk joins stay bulk, one `ps` for the whole table. Leg 101 of
  `tools/test-test-run.sh` reads the three from the script under test and requires one string for
  the harness shell and each of its ancestors, tells a changed `ps_token` and a changed bulk join
  apart (its negative controls), and counts the lines that read a start time: a new renderer
  fails it until it is pinned there. `lock_census`'s own bulk join is self-checking (a pid is
  paired only while `census_same` holds on the joined token). `tools/per-test-cap.sh` renders
  its own tokens, compared only against itself.
- **An embedded Perl program holds no single quote** (gh-ocannl-1945). `tools/test-run.sh` and
  the files it sources carry their Perl as `NAME_perl='` … `'` assignments, text `bash -n` never
  reads: a `'` inside one ends the string where it stands. A lone one usually breaks the shell
  parse; a pair around a word (`sub 'census_token' {`) leaves a script `bash -n` accepts and a
  program `perl -c` accepts too, the shell having joined the pieces back. So leg 1b of
  `tools/test-test-run.sh` requires each program's text to be quote-free (write `\x27` or `q{}`
  in the Perl), then compiles the value the shell assigns with `perl -c`, and requires every
  `$NAME_perl` the script expands to be one of the programs it checked. Run
  `tools/test-test-run.sh --only '^1b:'` after editing one. The inline `perl -e '...'` programs
  inside functions are not covered.
- **A cancelled run settles inside the fleet wrappers' 10 s grace** (gh-ocannl-1808).
  `fleet-worker.sh execution slot` and `bg-run.sh stop` send KILL 10 s after their TERM, so a
  supervisor that spent 5 s reaping and then up to 30 s in the slowest-actions reader was KILLed
  with no `exit:` sentinel (the run reads as died) and with a session-escaped dune descendant still
  holding the worktree lock (gh-ocannl-1727's incident). On INT/TERM/HUP the supervisor now reaps
  on a short grace (2 s TERM, 1 s for a KILLed leader, 0.5 s for the corpses). It then KILLs every
  other holder of the lock, found by the census it shares with `stop` (the bullet above; never
  itself or its launcher), and only then writes the verdict.
  Next it closes fd 9, and last, best-effort, removes a `fresh` run's build directory (renamed to
  `build.discard` before the verdict) and runs the reader on a 3 s bound. `wait` can digest such a
  run before its slowest-actions record lands. A cancellation that arrives while a normal finish is
  under way changes no verdict, but cuts the reader short (`slowest-skipped`: cancelled while the
  reader ran). The cap's own expiry (142) keeps the long order, since no wrapper is counting down
  then. Only a `run`/`start` supervisor runs the census (`OCANNL_TOOL_TESTRUN_LOCK`), because a
  repeat coordinator also holds the lock. `tools/test-test-run.sh` leg 97 TERMs a run during
  `_resolve`, during a TERM-ignoring dune, and during a `fresh` run, each with an escapee, and
  HUPs the launcher of a foreground `run` during that dune (relayed as TERM; gh-ocannl-1906). It has
  two controls: one cuts the census (the lock stays held) and one restores the reader-first order
  (no verdict within 10 s). Leg 74's fixture copies cut the census too. It is a second defence
  against the unmasked fork child, and leaving it in would let that leg's legs pass without the
  mask.
- **A run directory says which source it ran, as of the launch** (gh-ocannl-992): beside `cmd`,
  `cap`, `wt`, `runs`, `log` and the verdict, `run`/`start`/`repeat` record `head` (the checkout's
  HEAD commit) and `dirty` (its `git status --porcelain`: empty when clean, else one line per
  uncommitted path, untracked included). Before them a run's revision could only be inferred — from
  the worktree's HEAD at reading time, which a reset or a later commit moves and a
  commit-time-vs-`cmd`-mtime guard cannot tell apart (ludics-lite#158's review), or from the
  worker's transcript (staging#762 rebuilt its runs' pre/post commits and dirty instrumentation that
  way). Evidence cites a run as tested at `head` only when `dirty` is empty; a nonempty one is a run
  over uncommitted edits, whatever the report says. The record is SOURCE only: ignored files (a root
  `ocannl_config`, which a test can pick up through the ancestor search) and the environment
  (`OCANNL_*`) are configuration, and a report that depends on them states them. Both files are
  optional: written only where the script's root is a checkout's TOP level (a fixture copy under
  some checkout would otherwise record that checkout's HEAD), `dirty` first, so `head` means both
  exist; git's repository-selecting variables (`--local-env-vars`: `GIT_DIR`, `GIT_WORK_TREE`, ...)
  are dropped, untracked files are listed whatever `status.showUntrackedFiles` says, and the status
  runs with `GIT_OPTIONAL_LOCKS=0`, so the launch never rewrites the index. HEAD is read on both
  sides of the status and the pair retaken while it moves, so a commit landing in between cannot
  pair the old commit with the new tree's clean status. An edit made after the launch — during a
  fleet slot's wait, or while dune runs — is in neither. Legs 59–63 of `tools/test-test-run.sh` pin
  clean, dirty, an inherited `GIT_DIR`, the absent cases and that mid-record commit.
- **A checkout with unmerged paths is refused: `UNMERGED REFUSED`, exit 2, nothing ran**
  (gh-ocannl-1991). Mid-merge or mid-rebase, HEAD is one side of an operation still in progress, so
  a verdict's `source:` commit would name a tree that is not what ran — and `dune build @check`
  passing is no guard when the conflict sits in a file no rule compiles: a Metal run launched
  mid-rebase that way, its only conflict in a Markdown file (gh-ocannl-1954's hand-back). `run`,
  `start`, `fresh` and `repeat` refuse first of all, before the alias check asks dune anything,
  naming each path as the launch record's own line (`UU docs/notes.md`); `plan` says run would
  refuse. The refusal READS the `dirty` record above rather than asking git again, so it and the
  digest's `source:` line state one launch-time fact, and it has the record's scope (none outside
  Git or below a checkout's top level). Only unmerged INDEX entries count: a conflict resolved and
  staged runs, mid-merge, as any dirty tree, which is what the conflict drill under *Promotion and
  formatting* needs. `OCANNL_TOOL_TEST_UNMERGED=allow` runs the tree as it is, for the one drill
  that wants it — regenerating a golden that is itself the conflict, which `tools/promote.sh` then
  promotes and deliberately leaves unstaged — and that run's `source:` line counts the unmerged
  paths (`+ 2 uncommitted paths, 1 of them unmerged`), so it cannot be quoted as a commit's. Leg
  104 of `tools/test-test-run.sh`, which fails in five checks against the script before this
  (`--subject`).
- **Whether a run left anything to promote is dune's answer, recorded at the run's end**
  (gh-ocannl-1087): the run's last phase writes `promotions` beside `head`/`dirty` (what
  `dune promotion list` prints, one source path per line, empty when there is nothing) before the
  supervisor publishes `exit`, and the digest names those files. It used to infer promotion from the log, and
  four review rounds of staging#827 each found a shape it misread (colored and patdiff headers, a
  chosen `--diff-command`, a quoted `.corrected` stanza, a hunk above the scanned tail): dune's log
  is not a contract, its list is. Two traps in taking it: the list is on stderr before dune 3.22
  and on stdout since (the version picks the stream; a line that is not a path in the worktree
  voids the record), and asking recreates a build directory a `dune clean` just removed and
  rewrites dune's default trace, so a missing build directory is an empty list asked of nobody and
  the query gets a trace file of its own. The record also outlives dune's own list, which the next
  build in the worktree replaces — an unrelated failing one empties it, after which `dune promote`
  says "Nothing to promote" for a run whose digest offered promotion (staging#840). It is dune's
  list at the run's end, not the run's own diffs, so the digest reads it only under a pass or FAIL
  verdict (a refused slot ran nothing and found an older build's list), and the log inference
  survives only for runs with no record: another `--build-dir`/`--root` in the argv, an
  unreadable answer, a nearly spent cap, `repeat`, and runs from before. Legs 71-73 of
  `tools/test-test-run.sh` pin it; legs 68-69 now pin the fallback.
  Recoverable records also atomically publish `promotion-files` before the verdict (gh-ocannl-1154):
  numbered corrected bytes, originals, and a path manifest. `tools/promote.sh --from-run RUN [FILES]`
  accepts an exact selection from a finished record in the same worktree, refuses later source edits,
  normalizes goldens to LF and preserves the merge staging guard. Capture uses `dune promotion diff`
  with a literal copy command receiving the correction path: `show` adds a newline and is absent on
  3.20; the diff interface works for both staged intermediates and declared targets at the floor.
  Capture failure keeps the list but advertises no replay; every copy/query stays under the run's
  existing lock, signal handling and remaining cap. `tools/test-promotion-record.sh` owns the byte,
  recovery and refusal controls; `tools/test-promote.sh` owns the opposing floor-stream controls.
  Their version legs SIMULATE the floor with shims over the installed Dune; CI's `promotion-floor`
  job is what runs them against a real floor binary.
  Floor Dune also recomputes diffs during `list`, so both wrappers pin its `--diff-command=diff`: a
  presentation-only `DUNE_DIFF_COMMAND=-` must not hide registered corrections from recovery.
- **`cmd 2>/dev/null` does not silence a failed REDIRECTION.** The shell reports that before the
  command's own stderr redirection applies, so `read -r line <"$f" 2>/dev/null` prints
  `/proc/NNN/stat: No such file or directory` whenever the entry vanishes mid-scan — routine, not
  exceptional, for a glob over every process on the box, and for any reader asked about a pid that
  is supposed to be gone. `{ read -r line <"$f"; } 2>/dev/null` is the spelling that suppresses it.
  All four `/proc`-reading shell tools here (`tools/test-run.sh`, `scripts/setup-ocaml-env.sh` and
  both hand-run harnesses) use the grouped form; the plain one leaked the message into a caller
  that captured stderr.
  The former corpses-only sentence had no constructible state: its branch opened only for a leader
  that passed `group_verified`, while `proc_alive` rejects a zombie leader and a live leader is
  itself running work. It was deleted in gh-ocannl-832; the surviving-processes sentence now warns
  that the group may hold only unreaped exits, and the harness pins that combined wording without a
  copy of the shipping script whose predicate is artificially forced.
- In an errexit shell test, `! cmd` is not an assertion. Bash exempts a command whose value is being
  inverted, so `! grep -q 'must not appear' "$out"` runs, returns 1 and the script carries on — the
  negative half of a test can be entirely inert while reading as covered. Spell it as a function
  whose body uses `if`, so the command errexit weighs is the CALL and the ERR trap names its line,
  and have it print what matched: `$BASH_COMMAND` from inside a function names the body, not the
  pattern. Same shape as the `p_all`/`p_none` rule for `Verdict` claims — a check that cannot fail is
  worse than a missing one, because the golden and the roster both count it. Its sibling is a bare
  `[ A ] && [ B ]`: errexit exempts every operand of an `&&` list but the last, so the pair checks
  only `B` and is silent exactly when `A` — usually the point of the assertion — fails
  (gh-ocannl-1023, `cancel_sweep`'s readiness check). One predicate per statement, or end the list
  with `|| die …` / `|| return 1` (`|| rc=$?` to capture it; a `rc=$?` on the next line is refused). `shell_scripts_parse` refuses both shapes where errexit may be on and nothing consumes the
  status. `Shell_context` decides both: it reads the compound tree, carries errexit through it in
  source order as a may-be-on bit (its imprecision can only refuse, never pass), and treats a
  condition and the LAST statement of a function body or subshell -- also through a final group,
  branch or arm -- as consuming the status: a function's last command is its return value
  (gh-ocannl-1220) -- unless every call of it the file shows discards that value (`f | cat`,
  `f &`, or a statement nothing consumes with errexit certainly off; gh-ocannl-1266). A status
  something can overwrite before its consumer is not consumed: a loop body's tail (the next
  iteration) and a `;&`/`;;&` arm's (the arm it falls into). A function body that may hand errexit
  back on to a caller that had it off -- also the `set +e … set -e` save-and-restore -- makes the
  whole file read as errexit-on, since the scan does not follow errexit through calls; the golden
  names such a file and its functions on a `read with errexit on from its first line` line. A
  `set -e` in a `( )` subshell, or in a pipeline element or background job that cannot run in the
  body's shell, does not count: a `return` or `break` beside it leaves only that subshell, so a
  subshell-scoped errexit is the way to want errexit inside one step of a function (gh-ocannl-1577;
  the scan reads shell structure, not semantics, so this is a rule about where the `set -e` sits).
  Inside a function body capture a status with `|| rc=$?` rather than toggling errexit around it. Its module header states what it reads, refuses and leaves out (loud and
  silent). `Errexit_execution_controls` measures each context rule: it runs the script under the
  host's bash (3.2 on macOS, 5 on Linux) and again with a plain `false` in the assertion's place,
  and a rule bash disagrees with fails there -- add a row when the reader learns a context (a row
  needing syntax bash 3.2 lacks, like `;&`, is gated on the host's bash and pinned scan-side by an
  arm fixture). A structure the reader cannot balance is refused as unsupported in a script that may
  turn errexit on. Both arms read through
  `Shell_lexer.numbered_spliced_lines`: it carries multiline quotes and literal outer heredoc
  delimiters, excluding their bodies, and joins continued condition headers through `then`/`do`.
  Continued unquoted heredoc bodies are refused because removing their escaped newlines can change
  delimiter recognition; quoted bodies retain those newlines. Its header names the supported
  delimiter grammar and its explicit refusals (gh-ocannl-907).
- macOS's bash 3.2 cannot parse a comment holding an apostrophe inside a process substitution
  (`< <( # this tree's ...`): the error (a "bad substitution", no closing parenthesis) comes only when the
  line RUNS, never from `bash -n`, and a fail-open caller then reads it as an empty answer
  (gh-ocannl-1448's census did, and only leg 89 noticed). Put such a body in a function and
  substitute the call.
- The third inert shape, which no scan refuses: an assertion INSIDE `$(…)`. Outside POSIX mode
  bash clears `-e` in a command substitution (unless `shopt -s inherit_errexit`, which macOS's bash
  3.2 lacks), so `out=$(check_a; check_b)` runs on past a failing `check_a`; only the last command's
  status leaves, and only through a bare `var=$(…)` — as an argument, or under `local`/`export`,
  even that is discarded. Capture inside the substitution and assert outside it.
- **No credential reaches dune from a runner** (gh-ocannl-1280). Dune writes the environment of
  every process it spawns into `_build/trace.csexp`, inside the checkout, and the fleet exports its
  GitHub PAT as `GH_TOKEN` into every session — so every build copied it into a tree agents grep and
  mine, from where any transcript or log can carry it on. `tools/credential-env.sh` holds the one
  deny-list (`GH_TOKEN`, `GITHUB_TOKEN`, `GH_ENTERPRISE_TOKEN`, `CLAUDE_CODE_MESSAGING_TOKEN`,
  `*_TOKEN`, `*_API_KEY`) and prints it as shell TEXT, evaluated where dune runs and AFTER any opam
  environment is applied, since a switch's environment updates could set one too:
  `tools/test-run.sh` scrubs at startup and again in `select_dune` after sourcing
  `tools/opam-env.sh`; `tools/promote.sh` and `tools/dune-quiet.sh` after their own opam-env
  fallback; `tools/api-drift.sh` and `tools/ppx-output-diff.sh` at startup; `tools/ci-shard.sh` in the subshells of its `dune show`
  calls; `tools/fmt-check.sh` at startup and inside each `opam exec` of its default command (the
  formatter, then the scans); `tools/sweep.sh` starts
  every leg's dune as `opam exec -- sh -c <scrub; exec "$@"> dune ...` (`switch_dune_cmd`), in the
  text the far side of the ssh runs, whose login environment holds that box's own token;
  `tools/machine-verify.sh` passes the text to the far side, which applies it the same way inside
  every `opam exec` and nowhere else (so git's fetch keeps its credential). The scrub fails CLOSED:
  listing the environment (`env`, its one external step) is a checked operation, lines are cut by
  parameter expansion rather than a parser that could fail silently, `env`/`unset`/`printf` go
  through `command` (an inherited function cannot answer for them), a deny-listed name that is not a
  shell identifier refuses, and a second listing refuses anything deny-listed still present (a name
  holding a newline, which the line-wise reading cannot see whole). The boundary is those runners
  and the `tools/` wrappers the agent workflow sanctions; a harness staging one derives what it
  sources from its `. tools/...` lines. Outside it -- a bare `dune`, or a one-off driver such as the
  `benchmarks/*.sh` measurement scripts -- dune records whatever the session exports, as the issue
  scopes it; the fleet-side fix for that is not exporting the token into every session.
  `env_var_deps` reads the same line and refuses a stanza declaring a match as `(env_var ...)` or
  reading one through `%{env:NAME=...}`, and a test source reading one by name (`Sys.getenv`).
  Nor do the runner's own variables reach dune (gh-ocannl-1448): the supervisor deletes
  `OCANNL_TOOL_TESTRUN_BG`, `_RD`, `_OWN` and `_RUNNER` from its environment before it forks, and
  a phase of the run that needs its directory (`_resolve`, `_probe`, `_direct_plan`) is handed it
  as `env NAME=VALUE` on its own command line and unsets it on entry. A fixture that must tell the
  runner's own `--trace-file` apart therefore recognises it by where it points -- a directory
  holding a run's `wt`, `cap` and `log` -- not by an environment variable naming the run (leg 90
  of `tools/test-test-run.sh`; leg 84's replay compares an environment with no exemption for
  them).
  Pinned by leg 79 of `tools/test-test-run.sh` (a fixture dune's environment, the opam environment
  `select_dune` sources, `tools/dune-quiet.sh` run directly, and a real dune's trace where one is on
  PATH), `credential_free` in the sweep harness (local and remote legs, the remote one with
  credentials the fake switch sets, and a mutation control), the
  `credentials-ssh`/`credentials-switch` cases of `tools/test-machine-verify.sh`, the credential
  legs of `tools/test-fmt-check.sh`, `tools/test-promote.sh` and `tools/test-ci-shard.sh`, and
  `env_var_deps_control`; each has a negative control with the scrub cut out.
- A child that publishes a value for its parent to poll — a pid, above all — writes a sibling and
  renames it into place: `open(path, 'w')` creates the name EMPTY before the write lands, so a
  parent polling `exists()` reads `''` (gh-ocannl-1041, a per-PR-matrix flake). The benchmarks'
  Python tests go through `publish_pid` (`benchmarks/test/test_cell_group.py`), and a test there
  fails any pid written in place. The shell harnesses poll with `[ -s file ]` instead, which holds
  only because a pid lands in one `write`.

## Promotion and formatting

- **Promote through `tools/promote.sh` during a merge, on every platform.** Promotion writes the
  WORKING TREE; `git commit` during a merge takes the INDEX. So a golden promoted after its `git
  add` is committed with its PRE-promotion content, and nothing local objects — every later `dune
  runtest` reads the working tree and passes — while CI builds the committed tree and fails on the
  golden diff. It cost ~90 minutes on staging#487, and it is invisible by construction: the
  usual conflict drill (resolve, `git add`, run the suite, promote what moved) puts the promotion
  on the wrong side of the `add` every time, and `git status` shows the file staged, which is what
  you were checking for. `promote.sh` closes it: mid-merge it takes `dune promotion list` BEFORE
  applying (afterwards nothing pending is left to name), reading both streams for the pre-3.22 floor,
  then stages exactly what it promoted and says so. A promoted golden still UNMERGED is deliberately NOT staged and is warned about with the
  `git add` that would accept it — staging one records a conflict resolution, which is the caller's
  call and not the script's. Outside a merge the index stays untouched.
  Every invocation lists pending promotions before applying them, and reports existing backend
  golden siblings selected by a literal `(diff ...)` / `(diff? ...)` first argument in the
  destination directory's `dune` or `dune-file` (`%{read:...}`, gh-ocannl-2073). This covers
  backend names and emitted-code extensions without naming either selector file.
  Each sibling is marked `promoted` or `NOT promoted` for this invocation, including when replaying
  selected saved-run corrections. The test-run digest prints the same report below its recorded
  promotion list, using `pending promotion` / `NOT pending promotion` and the recorded worktree.
  Both paths share `tools/dune-golden-families.sh`, including the fail-closed Dune reader also
  used by test-run's logical alias-directory resolution. Unreadable or unsupported Dune forms
  (`#;`, `#|`, includes, unbalanced lists) produce an explicit unavailable notice. This is an advisory reminder to check other backends, not a
  claim they need edits or ran successfully. The report enumerates existing local files matching
  the selector's literal prefix and suffix; includes, escaped filenames, other interpolations
  and nonlocal selectors are outside its boundary. No backend list is duplicated.
  `test-promote.sh` leg 10 exercises selected, whole-list and saved-run promotion, an arbitrary
  backend, an extension selector, an earlier unrelated edit, a genuinely pending non-family
  golden behind a commented selector, and a cut-report negative control. Leg 11 covers unsupported
  Dune forms; `test-promotion-record.sh` checks the default digest from another checkout and a
  cut-digest negative control.
  The merge is detected with `git rev-parse --verify MERGE_HEAD`, never by testing `.git/MERGE_HEAD`
  as a path: in a linked worktree `.git` is a FILE and `MERGE_HEAD` lives in the per-worktree gitdir.
  `tools/test-promote.sh` is the hand-run harness — on no dune alias, since every leg runs `dune` on
  a throwaway project and a dune nested inside `dune runtest` brings its own lock, its own `_build`
  and the outer run's `DUNE_*`. It copies the working-tree script into each scratch repository,
  which is also what aims it: `promote.sh` resolves its repository as `dirname $0/..`, so where the
  copy sits is the repository it acts on. Its leg 1 is the negative control that the trap
  reproduces at all (a bare `dune promotion apply` in the same scenario commits the stale golden) —
  without it, leg 2 would pass if `git commit` merely picked up the working tree, and the guard
  would be untested. Measured: legs 2, 5 and 6 all fail against the pre-guard script.
- Formatting is a gate on the PR path, and its mechanics are worth knowing before the first red:
  a fmt-fix push costs a CI round AND a review round (gh-ocannl-938).
  `ci.yml`'s `fmt` job is separate from the build matrix on purpose: `ocaml/setup-ocaml` plus an
  explicit install of the ocamlformat release `.ocamlformat` pins, followed by `tools/fmt-check.sh`,
  which runs `dune build @fmt` in a switch WITHOUT the project's dependencies (`@fmt` compiles
  nothing), so it answers in a few minutes while the twenty-minute build is still queued. The same pin is what
  `dune-project`'s `(ocamlformat (and (= …) :with-dev-setup))` installs locally through
  `opam install . --deps-only --with-dev-setup` — a plain `--deps-only` never installs a
  `with-dev-setup` dependency, which is why a fresh switch has no ocamlformat until asked, and
  `scripts/setup-ocaml-env.sh` reports both that and a version drifted from the pin at session
  start. The wrapper also fails on ocamlformat's softer `Invalid documentation comment` warnings,
  which otherwise leave `@fmt` green and bury a real formatting diff in their output; a nonzero dune
  status still wins unchanged. That warning is printed only when ocamlformat's action RUNS, so the
  wrapper passes `--force`: after a `dune fmt`, every file that was already formatted has an
  up-to-date action, and a plain `dune build @fmt` replays nothing and passes a tree CI's fresh
  checkout rejects (gh-ocannl-1155; measured on this repository: the unforced gate exited 0 on a
  planted invalid comment the forced one rejects, for about two seconds more). The wrapper's last
  line is always a `fmt-check: PASSED` or `fmt-check: FAILED (exit N)` verdict; read that line or
  the exit status, and never judge a run piped through `tail`, whose status is `tail`'s (the
  slip behind staging#910's red). `tools/test-fmt-check.sh` pins the clean, warning, and
  formatter-error outcomes separately, each ending in its verdict line, and, where opam has dune
  and ocamlformat (CI's `fmt` job runs it after the install), runs a real project through `dune
  fmt` and then the wrapper, with the unforced build passing the same tree as its negative control.
  setup-ocaml's switch is local to the checkout: `opam exec` from a scratch directory exits
  50 (no switch) unless the harness first exports `OPAMSWITCH=$(cd <repo> && opam switch show)`.
  Resolve the switch before leaving the checkout, as `test-fmt-check.sh` does (gh-ocannl-1178).
  The default command then builds `@test/operations/scans` (gh-ocannl-1321) and prints how long
  it took. Measured on mac-studio: about 1 s when nothing changed, and about 30 s after a
  test-source edit, its slowest action `env_var_deps` at 10 s beside `codegen_text_inventory` --
  `env_var_deps` was 47-60 s until gh-ocannl-1425 retired its per-key derivations. `OCANNL_TOOL_FMT_CHECK_SCANS=0` skips the step and
  says so in the verdict line; CI's `fmt` job sets it, since its switch holds no project
  dependency and the build matrix runs the scans anyway. A command given as arguments runs alone,
  which is the form `tools/test-fmt-check.sh` drives with a fake switch: the scans build after a
  clean formatter run only, their failure is the verdict, and any other setting is refused. The
  script takes no options: an argument starting with `-` is refused with the usage (exit 2),
  where `fmt-check.sh -j 4` once ran `-j` as the command and exited 127. On a fleet box the
  default command takes one of the box's run-time correctness slots for both builds itself,
  as `tools/test-run.sh` does and through the same probe and candidates (gh-ocannl-1519): before
  that, a worker ran it outside any slot while a measurement held the box, because the format
  rule's last step reads as formatting, not as a build. It asks for `--cpu`
  (`@test/operations/scans` is on `Slot_kind.config_free_aliases`, `@fmt` links nothing), waits
  `OCANNL_TOOL_SLOT_WAIT` (600 s), and reuses an enclosing slot. The check reruns itself under
  `execution slot` and reports back through a file (`fleet_slot_rerun`, shared with the harness
  runner), so a slot that never ran it is `fmt-check: FAILED (exit 75): SLOT REFUSED`, nothing built, while a check the slot ran keeps its own
  verdict line as the last and its recorded status as the exit status, whatever the wrapper's. A command given as arguments takes no slot (the harness's form;
  wrap a caller's own formatter yourself), and `OCANNL_TOOL_FLEET_WORKER=none` turns it off as
  for the runner; `tools/test-fmt-check.sh` drives the slot through a fake fleet-worker.sh.
  Two files a formatter cannot handle are refused at the site rather than
  discovered in CI:
  a misplaced doc comment (ocamlformat declines the whole file) is a compile error under the root
  `dune`'s `-w +50`, and a ppx-expectation golden is in `.ocamlformat-ignore`, which
  `ocamlformat_ignore_scan` keeps in correspondence with `test/ppx/*_expected.ml`. The one
  interaction that needs an ORDER is with goldens that embed a `file:line` through `~here`
  (gh-ocannl-672): a reformat moves lines, so format first, then run the tests, then promote — a
  promote made before the format can leave the promoted block unformatted and the two chasing
  each other. Why a gate and not an off-branch sweep, since the repository ran one for a month: a
  repo-wide reformat commit conflicts with every branch in flight, so it could only land in quiet
  periods (which a wave of PRs never leaves), and it needs its own format-check-test-promote
  convergence loop for those goldens; on the author's branch the promote round already exists
  whenever such a file was edited, and formatting adds nothing to it. The sweep's commits remain
  listed in `.git-blame-ignore-revs`. Because master is always clean, `dune fmt` from a branch
  rewrites only the author's own files.

## Windows portability

- Two Windows C-runtime formatting differences make hand-formatted floats non-portable in goldens:
  it prints 3-digit exponents (`e+018` where Linux prints `e+18`), and it rounds representable
  decimal ties away from zero where glibc rounds to even (`%.1f` of `2.25` prints `2.3` there,
  `2.2` on Linux). `Ir.Ndarray.concise_float ~prec` normalizes the exponents, OCaml's `%h`
  hex-float format sidesteps decimal rounding entirely, and tie-free test data sidesteps it too;
  `test/support/test_utils.ml` packages the rules — `hex_float` and `set_binary_stdout` are
  portable by construction, while `print_float`/`print_floats` delegate to `concise_float` and so
  still need tie-free inputs. The line endings are the other half: `.gitattributes` pins
  `*.expected` and `test/ppx/*_expected.ml` to LF, and PowerShell's `Set-Content`/`Out-File` write
  CRLF, so goldens are edited with bash tools and promoted through `tools/promote.sh`, which
  strips CRLF on Windows.
- Three more Windows facts, each of which makes POSIX-shaped code silently wrong rather than broken,
  all measured on a stock Windows 11 box while making the scheduled sweep green again (gh-ocannl-588):
  - `Unix.sleepf` cannot sleep for less than the system timer tick. A request below 1 ms truncates to
    NO sleep at all; everything from 1 ms up costs a full 15.6 ms. A budget written as "N turns of a
    short sleep" is therefore not a duration there: `atomic_file_race`'s "20,000 × 0.5 ms = ten
    seconds" was a sub-second busy-spin that also burned the core its peer domain needed, and
    `Atomic_file`'s "8 attempts, 2 ms apart" was anywhere between 16 ms and 110 ms. Write every wait
    as a DEADLINE in seconds — the tick is machine-wide and anything in the session can move it to
    1 ms with `timeBeginPeriod`, so even a count of full-tick sleeps is not a fixed budget — and
    measure that deadline with `Mtime_clock`, never `Unix.gettimeofday`: a bound on a DURATION that
    a wall clock can move under is cut short or extended by any NTP step, manual correction or VM
    resynchronization. `arrayjit.utils` links `mtime.clock.os` for exactly this.
  - Neither `open_in`/`open_in_bin` nor `Unix.openfile` asks for `FILE_SHARE_DELETE`, so an open
    reader and a `Sys.rename` over the same target refuse EACH OTHER with `EACCES` — arriving as
    `Sys_error "Permission denied"`, and from the rename with no filename in the message, which is
    how a commit refusal is told from an `open_staging` one. Both refusals are transient and both are
    retried, but a reader that reopens the target as fast as it closes it leaves no window at all and
    then no retry budget wins: measured, one or two publishes in six hundred are refused on every
    poll for a full second. Hence `atomic_file.mli` states the consequence rather than promising
    against it — on Windows a publish MAY be refused — and a test of publication under load must
    claim what is true (every publication committed or was refused, none went missing, refusals stay
    rare) instead of claiming liveness the platform does not provide.
  - `MAX_PATH` caps the whole path at 260 characters, which is a different budget from the 255-byte
    per-COMPONENT limit. A fixture named at the component limit has no directory it fits in inside a
    build tree (`_build/default/test/operations` alone spends ~65), and every open of it fails with
    ENOENT. Gate such a leg with `Verdict.skipped ~aggregation:`Environment``, and make the gate a
    PROBE rather than `Sys.win32`: what is capped is the path this run actually got.
- Windows caps a whole COMMAND LINE at 32,767 characters, and older repo-wide scans hand every file
  they read to their executable as an argument, so they grow toward it with the repository. Past it
  `CreateProcess` fails and dune reports `Error: CreateProcess(): No such file or directory` — an
  error naming neither the length nor the executable, on whichever scan the last few merges pushed
  over (three had crossed it and a fourth was 255 characters short when this was first hit). The
  scans with explicit glob lists therefore pass them as `@<path>` RESPONSE FILES: the rule writes
  the list with dune's own `(echo "%{deps}")`, which runs inside dune and spawns nothing, into a second target of
  the same rule, and `Test_scan.Scan_argv.expand` splices it back in where the `@` argument stood.
  A new whole-tree scan should instead use `Source_inventory`; `env_var_deps` also knows the older
  response-file shape — a target
  the rule hands back to its own action needs no golden diff — and recognizes it by the `@<target>`
  reference rather than by the name, so a real output cannot be renamed out of the requirement.
  The one thing this transport gives up is a path containing WHITESPACE: `%{deps}` is space-joined
  and dune has no boundary-preserving expansion, where argv preserved boundaries. It is given up
  loudly — every word of a response file is a dependency dune materialized, so `Scan_argv.expand`
  refuses a word that names nothing on disk rather than scanning two phantoms in place of one file.
  The repository has no such path today. `command_line_length_scan` (a `scans` member,
  gh-ocannl-1748) holds the rule on Linux, where no cap exists to fail: a dependency SET (a glob,
  `source_tree` or `package` reached through `%{deps}` or a named dependency) on a spawned command
  line is refused whatever its size today, because dune's globs also match generated files (about
  half of a `.filelist` is `.pp.ml` renderings) and no reading of the source tree bounds them —
  `verdict_ratchet` crossed the cap a day after measuring 32.5K characters (gh-ocannl-1720). Every
  other spawned line is bounded from its pforms and fails at 28,000 characters (7,000 for a
  `(system ...)` line, which dune hands to `cmd.exe`, capped at 8,191); a pform with no
  bound, an unjudged `%{read:...}` and an action head the scan does not place are refused too. A
  `(test)` stanza may declare no targets, so it writes its response file under `(no-infer ...)` and
  lists its own `<name>.exe` in `deps`, which `no-infer` stops `%{test}` declaring
  (`slot_kind_live` is the pattern). Leave that dependency out and the build does not fail where the
  mistake is: the action runs the bare executable before dune has built it, and `posix_spawn` fails
  with ENOENT naming `<name>.exe` (gh-ocannl-1748).
- A Windows checkout with `core.symlinks=false` (git's default there without Developer Mode) writes
  every git symlink as a small TEXT FILE holding the link target. `docs/research/` indexes two notes
  at the `docs/` root that way, so any scan that reads documentation sees short stubs, and `config_usage_scan` reports an exemption count drifting to
  zero. It is not a repository bug and must not be "fixed" in the exemption list: CI's Windows job
  enables symlink evaluation (setup-ocaml runs `fsutil behavior set symlinkEvaluation`) and sees the
  real files. On a hand-run Windows box, `git config core.symlinks true`, then delete and re-check-out
  those paths (`git ls-files -s | awk '$1=="120000"'`) before trusting any doc-reading scan.
- `os.kill(pid, 0)` is not a liveness probe on Windows and must not be used as one. CPython
  implements `os.kill` there as `OpenProcess` followed by `TerminateProcess(handle, sig)`, so signal
  0 KILLS the process it was asked about, and a pid that already exited raises `OSError`
  (`WinError 87`) rather than `ProcessLookupError`, which no POSIX-shaped handler catches.
  `benchmarks/cell_group.process_is_alive` is the portable form — a zero-timeout wait on a process
  handle, where `WAIT_TIMEOUT` means "still running". `signal.SIGKILL` does not exist there either;
  `os.kill` with any other signal is `TerminateProcess`.

## Dune mechanics

- **Do not delete a `(test)`'s `.exe.output` from `_build` by hand.** Dune can retain the
  successful action in its cache while the automatic golden diff sees no output, then reports
  that the `.expected` should be deleted. `--force` does not repair this target-producing action.
  Run `dune clean` and rebuild the test's alias through the runner; accept no golden deletion.

- `(copy_files ...)` creates PASSIVE rules: they do not fire just because you build a sibling target
  in the same directory — only when listed in that target's `(deps ...)` or requested explicitly. A
  rule consuming copy_files output must therefore declare it. And validate a `(mode promote)` target
  from a clean state (`dune clean && dune build @alias`): stale `_build/` intermediates can satisfy
  an undeclared dep, so an incomplete build passes while the artifact is wrong. Assert content
  (size, object counts), not mere existence.
- **A rule that hand-lists the files a script reads runs sandboxed, and a symlink sandbox is not
  one.** Unsandboxed, an unlisted file is present whenever another rule happened to copy it into
  `_build/default/<dir>`, so the omission passes everywhere but a fresh build (`benchmarks/dune`'s
  `test_orchestrate` missed its own `ocannl_config` that way, twice). `(sandbox always)` alone picks
  the symlink sandbox, which isolates nothing from Python: a symlinked script's RESOLVED directory
  goes on `sys.path`, and `Path(__file__).resolve()` leads back to `_build/default` — dropping a dep
  still passed under it, verified. `(sandbox always preserve_file_kind)` (copy or hardlink) is the
  setting; there dropping any one dep fails the run. Dune also plants an empty `.git` above every
  sandbox, so nothing inside one can read the enclosing checkout's repository: a test of a
  revision-stamping helper builds a repository of its own. `config_dep_completeness` requires the
  setting of every rule whose program it cannot place — a PATH interpreter handed a workspace file,
  a shell line (gh-ocannl-1359). Sandboxing `runtest-gh1002_kernels` surfaced the hazard live: it
  never listed `box_gate.py`, which `gh1002_cells.py` imports at its top, and passed because
  `kernel_attribution` copied it. Such a rule keeps its alias-only `(no-infer (progn … (diff …)))`
  form inside the sandbox: dune stages the diffed output out of it, so `dune promote` works as
  before, while splitting the run into a target-producing rule draws `bin_smoke_membership_scan`'s
  refusal of any target producer running an opaque command.
- **The environment reaches a test through its `ocannl_config` dependency** (gh-ocannl-1149). Dune
  reruns a rule for a variable only where the rule declares it, and nothing tied the hand-written
  `(env_var OCANNL_<KEY>)` lists to what the linked pipeline reads: review found one missing key per
  round, and every test compiling through `Context.compile` read dozens of keys it never declared.
  So `test/config/dune` runs `ocannl_ambient_env` under `(universe)` -- once per invocation,
  milliseconds -- writing `ocannl_ambient`, a sorted, escaped, comment-only record of the
  environment variables `Utils.classify_env_var` addresses to the configuration, and appends it to
  the shared configuration as `test/config/ambient/ocannl_config`, which every test directory
  copies. An unchanged environment gives a byte-identical file and dune's early cutoff stops there
  (checked: a test declaring no key reruns under a newly set `OCANNL_TF32_MATMULS` and is served
  from the cache on the next identical invocation). Left out: the `OCANNL_TOOL_`/`OCANNL_LOG_LEVEL_<MODULE>`
  namespaces (test-run.sh exports a fresh run directory in the former every run, which would rerun
  the whole suite), and `OCANNL_BACKEND`, whose per-stanza declaration-or-marker rule above stays,
  so switching backends between sweep legs does not rerun backend-pinned stanzas. A directory that
  checks in its own configuration (`arrayjit/test`, `profiles/`, `startup_streams/`) copies
  `ocannl_ambient` and lists it beside `ocannl_config`. `config_dep_completeness` requires one of
  the three -- the carrying copy as the nearest configuration, `ocannl_ambient`, or `(universe)` --
  of every site it requires a configuration of, and checks both producers in `test/config/dune`;
  `slot_kind_live` accepts a reader of the carrying copy as reading test/config's backend only
  while its producer has exactly the two-`cat` shape. The per-key declarations that predate the record -- and the derivations in
  `env_var_deps` that asked for them: named schedule values, the cc width, ambient guards' keys,
  `Generated.init`'s `build_files_prefix` -- are retired (gh-ocannl-1425), and `env_var_deps` now
  refuses an `(env_var …)` of any name `Utils.env_var_in_ambient_record` is true of, the predicate
  `ocannl_ambient_env` writes the record from. What a stanza may still declare is what the record
  leaves out: `OCANNL_BACKEND` (the rule above), a reserved `OCANNL_TOOL_*` a module reads by name,
  a tracing gate, and someone else's variable. Checked by hand at the retirement: with every
  declaration gone and both served from the cache, `OCANNL_VIRTUALIZE_MAX_VISITS=7` reran
  `profile_precedence`'s rule and its guard refused by name, and
  `OCANNL_GPU_FISSION_KEEP_MAPPING=false` reran `gpu_serial_lanes`, whose fission claims read it.
- **The host reaches a test the same way** (gh-ocannl-1038). A verdict can read a capability a
  backend probes on the HOST -- the rocWMMA header tree HIP's tile-MMA support requires, the
  dispatch/OpenMP runtime the cc backend's Grid loops link, the ISA and vector width its SIMD probes
  find, the CUDA toolkit tree -- and none of those is a file dune sees: a declared `(env_var
  ROCWMMA_PATH)` closed only the vector that had fired (gh-ocannl-1032), not a tree appearing at a
  path already searched. So `test/config/dune` also runs `tools/host_caps.ml`'s `--record` under
  `(universe)`, writing `ocannl_host_capabilities` -- `Context.host_capabilities`, each backend's own
  device-free probes, the same list CI prints as its host line -- and `ocannl_ambient_env` appends
  it to `ocannl_ambient`. It therefore reaches every rule the environment record reaches, with
  nothing declared per stanza; `config_dep_completeness` checks this producer beside the other two.
  It costs about a quarter second per invocation (mac-studio): dune gives each invocation a fresh
  `TMPDIR`, so the cc probe cache starts empty there, and the invocation's tests then reuse what
  this rule probed. Device properties (compute capability, device limits, the GPU's architecture)
  stay out: reading them opens a device, which a probe run on every dune invocation, CPU batches
  included, must not do. A new host-side probe belongs in its backend's `host_capabilities`, never
  restated in a test or a stanza. The daily sweep copies the record into each full-suite unit's log
  (`host-capability:` lines) and, through `tools/sweep-unit-state.sh`, says `HOST CAPABILITIES
  CHANGED` with the lines that moved when a unit's record differs from its previous verdict's -- a
  box that lost a capability shows up as a diff between sweeps, not only as a failing test.
- **An ambient-environment GUARD runs because the record reruns it.** A test that refuses to run
  when an OCANNL variable that would rewrite its golden is set — `startup_streams`,
  `profile_precedence`, `config_profiles` — reads those keys through `Utils.read_env_var`, which no
  commandline flag or config file can outrank. The guard only executes when dune reruns the rule,
  which the ambient record in its `ocannl_config` (or listed `ocannl_ambient`) makes happen on a
  change of any of them; every rule running the program needs that dependency, since dune
  invalidates each on its own deps. A `setenv` pin around the run is still the better option where
  it is available. The guard keys, the schedule pipeline's keys (gh-ocannl-1149) and the cc vector
  width (gh-ocannl-1320) were once each derived and paired with hand-written declarations; the
  record made all three redundant (gh-ocannl-1425).
- **A program that starts a search on cc pins the width or says why it need not** (gh-ocannl-1368).
  The ambient record makes a rerun at another width happen; this rule keeps the DEFAULT run off the
  host's auto-probed width, which on CI's x86 runners is 32 or 64 bytes at random and changed what
  a 0.2 pre-filter kept (gh-ocannl-1363). `env_var_deps` derives the population: programs naming
  `Autotune.tune` or a value of the framework library (`lib/`) that reaches it, transitively
  (`Train.tune_placements`, `Calibrate.stream`; the golden lists them), with a run the backend rule
  says can select cc or multidev_cc. Each either pins the vector FILE -- `cc_vector_bytes` and
  `cc_vector_registers`, since at a pinned width AVX2 and AVX-512 still differ in register count,
  which caps the register tiles seeded -- or its program stanza carries `; cc-vector-width:
  independent -- <reason>`. Per key and cc-capable run, a `--ocannl_<key>=` argument or a
  `setenv` around the run decides (it must set a value the key accepts, never `-1` or `""`); only
  without either does the initializer count, and then only a call of a BLESSED helper
  (`vector_file_pin_values`: `Fissioned_chain.small_menu`/`small_menu_envelope`) in a top-level
  item before any search -- a program's own `Hashtbl.set` of the keys is no pin. The blessing is
  held equal to a derivation (`Config_key_scan.stating_reach`: the support values whose bodies
  write every key), so a new pinning helper is blessed in review. Each program of a plural
  `(tests)` stanza answers on its own, and a support value reaching `Autotune.tune` starts a search
  as the entry point does. A program's runs are its `(test)` stanza's own plus every stanza, in
  ANY dune file, naming it in command position (gh-ocannl-1901): a metal-only `(test)` re-run on cc
  by a rule beside it, or by `test/operations/dune`'s bench-line rule from across the tree, answers
  as any cc run does. A stanza that can select cc (or names no backend, read as running nothing)
  whose command names no program the reader can place (a shell line, `env probe.exe`,
  `env %{read:runner}`, `cmd /c "probe.exe …"`, an unknown action) is refused rather than read
  as running none; a PATH tool handed only arguments provably naming no `.exe` (a script) is the
  stated boundary, what it launches being past dune's text. The narrowness is deliberate: three review rounds of proving arbitrary
  statements of a setting each found a new way to write one. The reason has to be TRUE: a claim on what a search seeds, keeps or
  crowns pins instead. A marker on a pinned program, a stale one, a second one and one on a rule
  are refused. A test pruning any cc search with an envelope gets the pin from
  `small_menu_envelope`, which states the vector file beside the envelope. The local proxy for both
  x86 widths is `OCANNL_CC_VECTOR_BYTES=32` and `=64` on the arm box, which the ambient record
  reruns.
- Dune roots at the OUTERMOST ancestor holding a `dune-workspace` (failing that, a `dune-project`)
  and ignores dot-directories, so from a worktree under `.claude/worktrees/` the main checkout wins
  and the worktree is invisible to dune: targeted commands fail with `Don't know about directory
  .claude/worktrees/...`, while a bare `dune build`/`dune runtest` quietly builds and tests the
  PARENT branch. `scripts/setup-ocaml-env.sh` writes a one-line `dune-workspace` at the worktree
  root, restoring it as the root with its own `_build`. The checkout it acts on is the CURRENT
  DIRECTORY's when that is a worktree of the script's own repository (equal git common dirs, with
  a `dune-project`), else the script's own: the hook runs as `"$CLAUDE_PROJECT_DIR"/scripts/...`,
  and in a desktop-app worktree session that variable names the MAIN checkout, while hook handlers
  run in the session's directory, so going by the script's location reported the main checkout's
  staleness and wrote no `dune-workspace` (gh-ocannl-1812; it prints `ok    checkout <dir>` when it
  adopts one). The step tests the ancestor DIRECTORIES
  rather than git topology, since a checkout can nest inside another checkout that is itself a
  linked worktree living anywhere, and `--git-common-dir` then names the primary checkout, not the
  one dune would root at. That file is generated per worktree and gitignored, never committed —
  being the outermost, a tracked copy at the repo root would shadow every worktree's and pin them
  all back to the parent (the script reports `FAIL` for a `dune-workspace` in any ancestor, which
  it cannot override from below).
  With it in place, `--root .` and `dune promotion apply` are no longer needed from a worktree;
  `tools/promote.sh` remains the Windows path, for the CRLF stripping, and the path for ANY platform
  mid-merge, for the staging guard above. Worktrees placed outside the
  repo need none of this, but see no `ocannl_config` on their ancestor path.
  The same hook also fetches `origin master` (bounded, best-effort: offline prints `skip`) and
  prints a `WARNING` with the commit count and the recovery (`git merge --ff-only
  refs/remotes/origin/master`, or a rebase when there are local commits) whenever HEAD is behind it — because a worktree is
  created from the MAIN checkout's HEAD, whose `master` only moves when someone fast-forwards it
  after a merge, so a new worktree can start dozens of commits stale (79 on 2026-08-22) and a
  full suite run then tests old code. Read the checklist before the first build.
  That section has a hand-runnable harness, `scripts/test-setup-ocaml-env.sh` — run it after
  editing the section; it is on no dune alias, since its `bounded` legs sit out watchdog timeouts,
  and CI runs it only in the toolchain-free `harnesses` job alongside `tools/test-test-run.sh`
  (the `group_alive` bullet above has the SKIP contract the two share).
  It copies the WORKING-TREE hook into throwaway clones under a `mktemp -d` (never touching this
  repository's refs or config) and covers the watchdog (TERM at the bound, KILL 5s later, the
  process GROUP, rc preservation, no orphans), the counting wording and its two recovery commands,
  the offline `skip` with the count taken as of the last successful fetch, a branch and a tag both
  named `origin/master` not displacing the tracking ref, `FETCH_HEAD` left byte-identical, and
  which SSH launcher git ends up invoking with or without the appended OpenSSH options, and which
  checkout a run from a nested worktree acts on when the project dir names the outer one. Both
  harness bugs it exists to prevent were live during staging#430's review rounds: a throwaway clone
  that silently tested the COMMITTED script, and a `run` helper that executed its label as a
  command. When adding a leg, add the negative control too — mutate the hook and check that leg,
  and only that leg, goes red. The harness found one bug on its first outing: `bounded` decided
  whether to wait for its watchdog with a `kill -0` on the command's process group, and `kill -0`
  counts a ZOMBIE as present — git's ssh child is one, reparented when git exits and not yet
  reaped — so a fetch that had already failed in milliseconds read as still running and sat out the
  whole 30s bound, on every session start with an unreachable ssh remote. Emptiness is therefore
  not a signal question: `group_alive` (the shared `scripts/process-group.sh`) reads process
  STATES, from `/proc` where there is one and from `ps -A -o pgid=,stat=` otherwise, and only a
  non-zombie member counts as work. Where the
  reaper is a PID 1 that does not reap — the ordinary container case — the zombie is PERMANENT, so
  the first attempt at this, a short retry loop around the same `kill -0`, would not have helped;
  that is the shape to keep in mind before reaching for a timing fix here again.
  The hook is RUN, never sourced: it sets nothing in its caller's shell. Sourced (workers did, from
  zsh, where `status` is a read-only special parameter and the hook's `exit` closed the caller), or
  read by zsh at all, it hands itself to a bash child and returns that child's status, after one
  stderr line saying so; the harness's leg 4 pins it from bash and zsh, with mutants lacking the
  hand-off (and restoring `status`) as its negative controls. CI's `harnesses` job installs zsh for
  that leg, since the image ships none; a Linux box without zsh reports it as an unlisted skip.
- **Setup follows mid-session directory changes** (gh-ocannl-1920). `.claude/settings.json`
  runs full setup on `SessionStart` (`startup|resume|fork`). Switch hooks use `--switch`,
  which only repairs the dune root and persists `opam env` when `CLAUDE_ENV_FILE` is available:
  no fetch, dependency checks or installs. The [hooks reference](https://code.claude.com/docs/en/hooks#cwdchanged)
  scopes `CwdChanged` to shell commands, so `PostToolUse:EnterWorktree` covers the worktree tool.
  `ExitWorktree` returns to the already-prepared original directory. `PostToolUse` has no
  `CLAUDE_ENV_FILE`; plain stdout from either switch event only reaches the debug log.
  Harness leg 6 executes the configured commands, checking the dune root, persisted environment
  and unchanged `origin/master` on switches; it does not probe Claude Code event delivery.
- **A merged session worktree retires with `--regenerable _build`.** ship-pr's
  `post-merge-cleanup.sh` (lukstafi/ludics-lite) archives a merged session worktree out of the way
  and refuses one carrying ignored data the base checkout lacks, since an archive would put that
  data out of sight rather than keep it. A worktree that has ever built carries exactly that: its
  `_build` (92M on one sighting), so every OCANNL worktree was refused until the flag existed.
  `--regenerable <dir>` names one top-level directory a rebuild recreates, and the helper deletes it
  before judging the worktree. The flag has no default and the helper learns no build system from
  it, so the name comes from here: `_build`. `_opam` is the same shape in a worktree that holds a
  local switch (the SessionStart hook creates none; it only suggests one when no switch is usable).
  Benchmark fixtures regenerate only at real cost and are never passed. What a value may name, and the helper's refusals, are ship-pr's SKILL.md, *After it
  lands* (lukstafi/ludics-lite#534).
- **Far-side wave worktrees also carry nested test artifacts** (gh-ocannl-2179).
  For the coordinator's close-out, use `fleet-worker.sh farside-close <wave> --regenerable _build --regenerable '**/log_files' --regenerable '**/build_files'`.
  Tests leave regenerable runtime logs and generated kernel/debug files outside `_build`:
  `git status --short --ignored` in a checkout that ran tests showed `arrayjit/test/log_files/`,
  `test/config/log_files/`, top-level `log_files/` and `build_files/`. Without naming them,
  far-side close-out preserves the worktree for its ignored data (85 of 86 worktrees in one wave,
  all in the next). The quoted `**/` form matches the named directory at any depth, including
  the top level; it belongs to `farside-close` alone. `post-merge-cleanup.sh --regenerable`
  still takes one top-level directory name per flag, as above. This set covers disposable
  test output, not arbitrary ignored data: leave local switches, caches and benchmark fixtures
  for separate judgment, and inspect `git status --short --ignored` before extending it.
- Dune tracks an environment variable only where a stanza declares it, and the tracking reaches
  further than the stanza: `dune rules test/operations/<name>.exe.output` shows the `(Env
  OCANNL_BACKEND)` dependency travelling from the `(test)` stanza's `(deps ...)` into the
  `.exe.output` rule dune generates from it, so `OCANNL_BACKEND=cuda dune build …exe.output`
  really does re-run the test on cuda. What it does NOT do is tell you it ran there: a
  backend-uniform golden (GPU legs announcing themselves on stderr while printing the same
  `<claim>: true` on stdout) makes the cuda `.exe.output` byte-identical to the cc one, which is
  how gh-ocannl-622 came to read a cc-looking file as proof the recipe was broken. It was the
  inference that was broken; the recipe holds for DECLARED variables, and gh-ocannl-628 is the
  hole that was real — the lowercase spelling `read_env_var` consulted first was declared nowhere,
  so `ocannl_backend=metal` decided the backend while invalidating nothing. gh-ocannl-652 closed it
  from the other end: the environment has ONE spelling, `OCANNL_<KEY>`, and setting a lowercase or
  dashed spelling of a known key aborts the run with a message naming the spelling that works, so
  the variable cannot quietly decide nothing either — on case-sensitive environments, that is:
  native Windows's case-insensitive environment makes the lowercase spelling the SAME variable,
  read normally (`Utils.env_names_case_insensitive`; `test/operations/config_var_spellings` pins
  both readings on every host), while a dashed spelling differs on every platform and stays fatal.
- `env_spelling_gate` is one gate per DIRECTORY because dune aliases are per directory, and it
  depends on `(universe)` so it reruns on every invocation — no suite comes back green with a
  rejected lowercase spelling ambient. `runtest` and `slow` are gated separately; a gate in a file
  that serializes on `/ocannl_training_test` must take the lock, since an unlocked action in a
  locked file is what the next training test gets copied from (the gate starts no pool). Hand-written
  per-test aliases depend on the gate explicitly, while dune's GENERATED `runtest-<name>` aliases do
  not — a targeted run of a `(test)` stanza can be served stale under a rejected spelling — and
  `env_var_deps` checks that every alias with test actions in a directory carries that directory's
  gate. A golden pair carries it on the rule making the diffed output, `(deps (alias
  runtest-env_spelling_gate) …)`, not on the diff's alias rule (gh-ocannl-1378): the gate still runs
  before the test, a cached output is never re-run for it (an alias dependency adds no digest; the
  gate's `(universe)` reruns only the gate), and `dune rules` shows a file rule's alias
  dependencies where it never shows an alias rule's, so `tools/test-run.sh affected` lists every
  pair when the gate changes. `env_var_deps` follows that route: an alias whose rule reads an
  output of a gate-dependent rule is gated. A rule that runs and diffs in one alias action (the
  `slow-`/`train-` shape) keeps the dependency on itself.
- Deleting a file target out from under dune is not a way to force it to re-run: `dune build
  <that target>` afterwards exits 0 having produced nothing (observed on dune 3.23.1 with
  `test/operations/<name>.exe.output`), and `-f/--force` does not rescue it — `--force` only
  re-runs actions attached to ALIASES. Either force the alias (`dune build --force
  @<dir>/runtest`), or run the built exe directly with its cwd set to `_build/default/<dir>`, which
  is exactly the environment dune gives it — the same cwd, hence the same `ocannl_config` search
  root, that makes `dune exec` unusable: the config search walks up from the invoking cwd, where
  the root `ocannl_config` is gitignored, so the program finds no config or `Context.auto` silently
  picks a GPU. The cause is that dune trusts its own digest
  database and never stats a rule's targets, so a hand-deleted one is recorded as built forever;
  that also rules out the two other reflexes, since touching a source changes no CONTENT digest and
  deleting `_build/.digest-db` does not restore the memo either. Every golden-diff rule now has an
  alias to force (`dune build --force @<dir>/runtest-<name>`, see below), though for a `(test)`
  stanza that only re-runs the diff, not the executable (next bullet); for a target with no alias
  at all the recovery is `dune build --sandbox=copy <that target>`: sandboxing changes
  how the rule executes, which invalidates the memo and re-runs it. `dune clean` works too and buys
  a full rebuild, which on macOS means every fresh executable queueing behind XProtect again. Worth
  knowing before it bites, because the failure is silent in the dangerous direction: the missing
  target leaves whatever `.actual` was there before, so a probe that only diffs the file reads
  green while nothing has run. The same probe has a second stale-reading trap: discarding the
  build's stderr (`2>/dev/null`) without checking its exit status — a FAILED build (say, a
  warning-as-error from a temporary edit) leaves the previous `.exe.output` untouched, and the
  stale file reads as a green probe; that turned a negative control into a false positive during
  gh-ocannl-554.
- **To genuinely re-execute an unchanged `.expected` test N times, use the harness's repeat
  mode**: `tools/test-run.sh repeat N build @<dir>/runtest-<name>` runs every iteration in its own
  freshly cleaned, cache-disabled build directory, keeps each stdout/stderr and exit status, and
  diffs the pairs (about 45 s per iteration for a 35 s test on an M4 Max, the rebuild included; 21-23
  s for `runtest-hello_world_op` at `-j 4` on mac-studio, nearly all of it the rebuild). The rebuild
  cannot come from dune's shared cache (gh-ocannl-1252, dune 3.24.2): the default
  `--cache=enabled-except-user-rules` stores nothing for this workspace, compiles included (a fresh
  `DUNE_CACHE_ROOT` held 0 files after building `arrayjit/lib`, and an iteration took as long under
  it as under `--cache=disabled`), while `--cache=enabled` also stores the test runs and restores
  their outputs instead of re-running them. A GPU batch's repeat has `run`'s two bounds (next
  section's GPU bullet).
  **`repeat --direct N @<dir>/runtest-<name>...` pays that rebuild once** (gh-ocannl-1377): a
  build phase builds and runs the named tests into the set's own build directory, then each
  iteration replays the test's own target-producing rules in the built tree
  (`tools/replay-rules.sh`, from `dune rules`' expanded action, in the environment
  `dune exec --no-build` reports for the context plus dune's per-action TMPDIR and
  BUILD_PATH_PREFIX_MAP) and runs `dune build --force` over the aliases, which re-runs the alias
  actions (the golden diff, a run-and-diff or inline-test rule) and compiles nothing. On
  mac-studio an iteration of `runtest-hello_world_op` took 7-8 s at `-j 4` (its inline-test runner
  is an alias action, so that is the forced dune alone) after a 29 s build phase, against 21-23 s
  per clean iteration. Every set compares the first replay with dune's own run of the same rules
  (the build phase's snapshot of their targets): a replay that agrees with itself but not with
  dune is `REPLAY DIVERGED`, never a stable test — it caught, while the mode was written, `echo`'s
  space-joining, a python's locale coercion exporting `LC_CTYPE`, and the `SHLVL`/`_` a shell's
  `exec` drops (so the build phase's dune is never exec'd). Refused before any iteration: a rule
  dune runs in a sandbox (the scans' `(sandbox always)` targets), an action outside the replayed
  forms (`system`, `bash`, a directory target), a test the build phase never ran (`BUILD FAILED`
  when the build itself failed); for those use the clean form. Not reproduced: the named tests'
  dune scheduling beside each other (a replay runs them in turn) and a fresh tree (files a test
  leaves beside its targets persist, as between two dune runs). `tools/test-test-run.sh` leg 84
  pins all of it against a real dune, with a mutant without the replay as the control that dune's
  `--force` alone never re-runs a target-producing test.
  For one fresh, TIMED run of named tests (a before/after wall-time comparison),
  `tools/test-run.sh fresh @<dir>/runtest-<name> [@<dir>/slow-<name>...]` hands dune a build
  directory of the run's own with `--force --cache=disabled`: the closure compiles there (some
  ten seconds at -j 4 on mac-studio), nothing the tests build lands in the worktree's `_build`
  (only the runner's own helpers, as for `run`), and the digest prints each named test's
  seconds from the run's dune trace. A named test none of whose
  processes is in that trace is refused as `NOT RUN` (exit 2) rather than timed at zero: the
  guard against every stale-green reflex below, and against the one that cost a timing baseline
  (gh-ocannl-staging#1154): `--cache=disabled` without `--force` serves an unchanged `@slow` rule
  from the memo, silently.
  For a single extra run there is a lighter move: change the content of the `ocannl_config` the
  stanza depends on — append a comment line, `dune build @<dir>/runtest-<name>`, delete the line
  when done (the file is tracked; `git checkout` it only when it carries no other local edits,
  since AGENTS.md also sends one-off configuration changes there). The rule's memo holds the
  content of the last BUILD, not of the file, so the appended value must be one that build never
  saw: deleting `# rerun 1` without building and appending `# rerun 1` again is served stale
  (verified), while a timestamp, or a counter that never restarts, is not. For the directories
  that `copy_files` it in (`test/operations`, `test/einsum`, `test/ppx`, `test/training`,
  `test/gpt2`) that file is `test/config/ocannl_config`; `arrayjit/test`,
  `test/operations/profiles` and `test/operations/startup_streams` depend on their own tracked
  `ocannl_config`, and an edit to the shared one leaves their rules untouched. Each new content is
  a new digest for the `<name>.exe.output` rule, so the executable runs again with no
  recompilation. This is what sampling a timing-dependent test needs (a
  re-roll-or-skip decision needed it, gh-ocannl-staging#764), and every other reflex
  fails silently in the green direction, verified on dune 3.24.2: `dune build --force
  @<dir>/runtest-<name>` re-runs only the alias's diff action, while the content-keyed
  `<name>.exe.output` rule that runs the executable is served from the memo; a comment appended to the test's `.ml` rebuilds it, but
  the compiled objects are byte-identical, so early cutoff serves the same `.exe.output` again;
  appending to the copied `test/operations/ocannl_config` instead fails with "Multiple rules
  generated" (it is the target of the `copy_files` stanza at the top of `test/operations/dune`) and leaves an untracked
  file to `rm`. `dune build --sandbox=copy @<dir>/runtest-<name>` does re-execute, but once per
  toggle of the sandbox mode, and each toggle re-runs every rule the alias reaches (over a minute
  for one 35 s test). Deleting `_build/default/<dir>/<name>.exe.output`, with or without
  `--cache=disabled`, is worse than useless: the rule stays recorded as built (previous bullet), the
  test's `diff?` then sees no output and every later build of the alias fails with "File
  `<name>.expected` should be deleted" until some real dep changes — the config append is also the
  recovery.
- Dune truncates a long stderr of a `(test)` stanza's run (`...TRUNCATED BY DUNE...`), keeping its
  head and tail. Verdict's teardown line, in that tail, names the first three failed checks
  (`FAILED: 5 checks did not hold: "a", "b", "c" and 2 more.`), so a truncated failure still says
  which claims to look at. To read a long per-row table, run the executable yourself inside
  `_build/default/<dir>`, where the rule left `ocannl_config`, under the fleet's `execution slot`
  when on a fleet box.
- **Three dune introspection answers say less than they look** (dune 3.24.2, gh-ocannl-1541).
  `dune show aliases` and `dune show targets` print their listing on STDERR and exit 0 after a
  load error (a dune file that does not parse, verified; an `(include)` of a missing file), the
  error on the same stream: read both streams and check every line's form, or compare against a
  listing known good, as `tools/affected-tests.sh` does. `dune rules` prints an `(:include)`
  dependency as the included file alone, never what the file adds (`dynamic-run` likewise). And
  `dune rules @ALIAS` expands the alias's DEPENDENCIES, helper aliases included, never its
  actions: dune evaluates an alias action lazily and records no dependency for it, so what an
  alias rule reads shows in no `dune rules` query. `affected` therefore rewrites each such rule
  into a file-writing twin (the `affected` bullet under Aliases and goldens).
- **A `(locks …)` name without a leading `/` is a lock of the stanza's own directory** — the same
  relative name in two directories is two locks. Verified 2026-10-31 on dune 3.23.1 and 3.24.2 (the
  repository's floor is 3.20; no changelog entry since 3.17, which made `(test)`/`(tests)` honour
  `locks`, touches how a name resolves) with two-second sleeper actions in a scratch project, two
  per directory in `a/` and `b/`, logging start and end times: under `(locks shared)` the two
  directories' actions overlapped while each directory's pair ran in sequence, and under
  `(locks /shared)` all four ran one after another; a `(test)` stanza behaved the same. `dune rules`
  prints no lock, so a run's timings are the only reading. So the training lock was two:
  `test/operations` (~30 actions) and `test/training` (~22) each spelled
  `(locks ocannl_training_test)`, and an operations action that runs training steps could start its
  OpenMP pool beside a `@train` integration's, the overlap every one of those stanzas' comments says
  the lock prevents. Every one of them now names the global `/ocannl_training_test`, and
  `env_var_deps` refuses any other spelling of it (bare, `./`- or `../`-qualified) in every dune
  file, and counts only the global one as taking the lock for the ambient-gate obligation. On the
  repository itself, six locked `test/operations` tests and six locked `test/training` ones built
  together (`-j 4`, cc, mac-studio, an explicit `--trace-file`) overlapped across the directories
  in 7 pairs (9 s) under the relative spelling and in none under `/ocannl_training_test`, with no
  overlap inside one directory either way. Serializing them did not lengthen the suite: on
  minix-amd-linux (cc, 32 threads, exclusive window, one `_build`, every config-reading test rerun
  by a comment line in `test/config/ocannl_config`), `dune build @runtest @train` took 869.7 s and
  874.2 s with the relative spelling and 781.4 s and 773.0 s with the global one, rounds ordered
  ABBA — about 11% faster, plausibly because two OpenMP pools that each claim every core no longer
  contend; the chain the lock makes is not the suite's critical path. Other locks in the tree (`cc_kept_library`, `schedule_conv_gemm`, …) are meant per
  directory and stay relative; `generated_lock_scan` already resolves both spellings as dune does.
  `env_var_deps` refuses any relative lock spelling written in two or more directories (a
  `(subdir …)` group counting as its own) that resolves to different paths there, naming each
  directory and suggesting `/name` (staging#1593); `../shared/name` from two sibling directories is
  one lock and passes, and a lock name built from a `%{…}` pform is refused as unreadable.

- Runtime test helpers live in `test/support/test_utils.ml`; repository readers live in
  `test/support/test_scan.ml` (gh-ocannl-2010). Keep `ppxlib` and `compiler-libs` on the scanner
  side: the compiler archive forces its units into every executable on its dependency path,
  even one using only a printer. `test_utils_linkage` compares the linked native units with the
  installed compiler archive's unit census, so this boundary has no hand-maintained byte limit.
  Compare clean builds with the same aliases before attributing disk growth to source changes;
  old build trees can contain only a subset of the currently declared executables.

## Verdict claims

- A claim sequence only some backends (hosts, configurations) can evaluate is written ONCE, as a
  leg (gh-ocannl-1703): `let leg = Verdict.leg ~when_ ~on ()`, then each claim as
  `in_leg ~leg label (fun label -> p label …)` (`in_legf` for a computed label) in code that runs
  on every backend, its evidence in `lazy` values only the `make` functions force. Where the leg is
  closed each claim is `skipped` under the same label, in the order the same code reaches it; a
  gate only the open side can compute is a `Verdict.sub_leg outer ~when_:(lazy …) ~on`. The two-
  branch form it replaced -- the evaluating branch's claims, and `List.iter claims ~f:skipped` over
  a hand-written list -- kept the golden's off-gate order in a second copy nothing tied to the
  first: `grid_launch_floor` appended two claims to its list while the CPU branch made them
  mid-sequence, and Metal, CUDA and HIP went red (gh-ocannl-1668); about thirty tests carried the
  same risk, through literal lists, binding lists or `List.nth_exn` indexing, and were migrated.
  `in_leg` holds its `make` to exactly one `<label>: <bool>` claim under its own label (no
  `pass_fail`, no silent `claim`, no second claim) and fails the run otherwise, since the closed
  side prints `<label>: true`; `verdict_quantified` pins both sides and each refusal. A single
  claim with one gate stays `gated` (or a one-label `if … else skipped label`); code outside the
  `make` functions runs off the gate too, so it computes only what is computable there. A leg's
  `~when_` is a GATE, read by the gate reader below (gh-ocannl-1794): it may not read what the
  leg's claims quantify -- `sub_leg cpu_leg ~when_:(lazy (not (List.is_empty (Lazy.force tiled))))`
  turned a seeder that lost its twins into two skips; gate on the invariant that decides whether
  the population exists (`schedule_conv_gemm`'s cvr leg reads `Register_tile.alternatives` at the
  site's own extents; `ptx_target_parity` reads that the backend is the one compiling PTX, and
  claims the PTX was recorded). The two-branch form is refused too: `verdict_ratchet` reads every
  `skipped` the resolver reads as `Verdict`'s, or a `let` binds to one (`Verdict_scan.skip_list_census`) and
  refuses one handed to a collection-module iteration, a callback or loop body that only skips,
  and a branch of an `if`/`match` applying two or more; the sites that predate the rule are
  `exempt_skip_lists` entries, each with its site count and reason. Both readers work under one
  BOUNDED model of what runs, stated once in `verdict_scan.ml`'s skip-list header (*The model*):
  a function value or `lazy` is deferred unless a collection function calls it back, and a
  construct outside the model is read conservatively -- no skip list, an uncompared leg -- never
  refused. Extending the model starts in that header.
- A negative control confirms the run REACHED its last row, not only that `FAIL:` lines appeared
  (gh-ocannl-1067): the claims it printed, passed plus failed, number the golden's lines, or the
  rows run under `Verdict.case`. A raise ends a run at the case that raised, and the rows after it
  are absent, not failed. `tools/mutation-run.sh` checks it (gh-ocannl-1083; its bullet below): a
  mutated run that stopped early or never ran exits 4 under `STOPPED EARLY` or `NEVER RAN`, not
  with the 1 that reads as "caught". gh-ocannl-1016's control read six FAILs as "the rest pass"
  while a raise had cut the run two rows short, invisibly: Verdict's teardown called `exit 1` from
  `at_exit`, which the runtime runs before printing the exception. The teardown now raises instead, and
  Verdict's uncaught-exception handler prints `STOPPED EARLY` and the exception (status 2).
  `Verdict.case label f` turns a raise into `<label>: the case ran to completion (raised …): false`
  and runs the next case, printing nothing when `f` returns, so wrapping is golden-neutral
  (`footprint_materialization` is the exemplar; `verdict_teardown` pins every ending). An `exit`
  inside a case, once a check has failed, is not a case failing but the run ending there, so
  Verdict prints `STOPPED EARLY: an exit inside case "<label>" …` for it (gh-ocannl-1084). The
  hand-built-IR `virtual_*` tests run their `case_*` rows through it too (gh-ocannl-1084), as do
  the multi-case `test_*` drivers that do not catch their own exceptions (gh-ocannl-1143). A labeled
  row helper owns the boundary, so future rows inherit it without repeating the label; a new
  multi-case test does the same. No scan holds it: the `case_` prefix is a local habit, and the
  wider population of `test_*` drivers often catches its own exceptions to print them.
- A claim is one row on each stream it prints to, and its stderr line is its stdout row behind
  `FAIL: ` -- `pass_fail`'s too (`FAIL: <label>: FAIL (<detail>)`; a bare `FAIL: <label>` reads
  exactly as a `Verdict.fail` message). `tools/mutation-run.sh` lists a run's failed claims by
  those lines' endings (`: false`, `: FAIL`, `: FAIL (…)`) and counts its stdout rows against the
  golden's. A `Verdict.fail` message prints its `FAIL: ` line on BOTH streams, so one ending like a
  claim is told apart by its stdout row and listed under `failure messages`, never as a claim; and
  a run's own `FAIL: ` rows (beyond any the golden has) count toward no golden row, so a run cut
  short after a long failure cannot pass for complete (gh-ocannl-1364). A label carrying a newline
  split its claim into a `FAIL: ` head that is no false claim and a tail that is a row the golden
  lacks. So Verdict renders every claim line, label and detail, through `Verdict.one_line`
  (a control character becomes a space), which keeps a computed label on one row; a literal label,
  or a `pf` format, written with a line break is refused by `verdict_ratchet`
  (`Verdict_scan.line_break_census`) -- a blank line before a claim goes in a `printf` of its own.
  `Verdict.fail` takes a message, not a label, and stays multi-line with every line prefixed.
- One claim surface, opened rather than copied. Every test that decides a verdict reaches the claim
  names through `open Verdict.Claims`; nothing in the tree rebinds them per file any more
  (gh-ocannl-815). The aliases the population used to carry were a maintenance defect with a
  demonstrated failure: `Ll_test` re-exported six of the names for the hand-built-IR tests, and the
  copy went stale — `pf`, `claimf`, `pass_fail` and the pairwise form were added to `Verdict` and
  never reached it. So a NEW combinator goes into `Verdict.Claims` and is thereby available
  everywhere; it does not get re-exported from a support library, and a test does not alias it.
  What stays qualified is what `Claims` deliberately excludes: the run-state readers (`any_failed`)
  and the backend-bound wrapper each backend-selecting test still builds for itself,
  `let skipped = Verdict.skipped ~backend:backend_name` — a partial application, not an alias.
  `verdict_ratchet` models the open explicitly, so helper-following still sees an unqualified `p`.
- `Verdict` gates a claim by exit status, and a claim whose LABEL is computed needed an entry point of
  its own: `Verdict.pf fmt … b` is `p` with the label rendered from arguments
  (`Verdict.pf "%s gradients match the oracle" leg ok`), and `Verdict.claimf` is `claim` the same
  way. The boolean comes last, after the format's arguments, so the call reads in `p name b` order
  and forgetting it is a type error rather than a silent no-op. Before they existed every
  computed-label claim was a bare `printf`, which exits 0 on `false` and lets the failure be
  `dune promote`d into the golden — the whole of gh-ocannl-624. `verdict_ratchet` now holds both
  shapes, its reader having widened twice: the separator vocabulary is `:`, `=` and `->`, since
  reading only a colon left the entire `… = %b` population outside a check written to catch exactly
  it; and a format may carry other conversions, since a computed label carries one by construction.
  That second widening costs the old escape hatch — a descriptive `%b` print can no longer excuse
  itself by interpolating what it describes — so a census row that ends on its boolean takes a named
  exemption, keyed by the format HEAD (everything up to the `%b`) rather than the whole format,
  because the whole format is itself the claim shape and a list of them would force the check to
  exempt its own file. Every exemption in the tree is a row whose assertion is claimed separately
  beside it through `Verdict.claim`/`claimf` on the same bound boolean; an exemption without that is
  one that should not have been granted.
- A quantified claim needs a non-emptiness guard, and the guard has to be the SHORTEST thing to
  write or it does not get written. `p "every seed spreads j" (List.for_all seeds ~f:…)` is `true`
  when `seeds` is empty, and the line it prints is byte-identical to one a hundred seeds passed —
  the gh-ocannl-601 hazard arriving through `Verdict.p` itself, invisible in the golden by
  construction (gh-ocannl-729). It bites hardest where the claim is worth most: quantified over a
  DERIVED collection — the seeds a family tree yields, the refutations a gate raises, the
  candidates that validated — where empty is a plausible regression rather than an impossible
  state. So `Verdict.p_all name xs ~f`, `p_alli` (the indexed form), `p_none name xs ~f` (the
  mirror: filtering an empty list
  also yields nothing, so `List.is_empty (List.filter …)` has the same hole), `p_exists`, and
  `p_empty name ~over:population derived` for the sites that keep the derived subset around to
  report it; a claim some host skips takes their gated forms, `gated_all`, `gated_alli` and
  `gated_exists ~when_ ~on` (the conventions note). A non-empty collection prints exactly what `p` prints, which is what let ~44 files
  convert with their goldens unmoved; an empty one prints `<claim> (empty): false`, and `?min:n`
  prints `<claim> (only 1 of 4): false`. Arrays go through `Array.to_list` rather than growing a
  second family. What stays on the unguarded spelling is the claim whose passing reading IS
  emptiness — "no candidate declines", "no key is undocumented", a scan over a tree that is usually
  clean — and those want a companion claim that the population was there at all, which is what the
  `p_empty ~over` form is. In short, the scan refuses a `List.for_all` (or `for_alli`,
  `for_all2_exn`) over a population it cannot prove non-empty, written as `p_all` (`p_alli`,
  `p_all2`) instead, and a bare `List.is_empty xs` claim, written as `p_empty name ~over:population
  xs` (or `p_none` for an `is_empty (filter …)`).
  `verdict_ratchet` enforces the guard mechanically, for a quantifier (`for_all`/`for_alli`,
  `for_all2_exn`, `is_empty`, a negated `exists`/`existsi`, in Base's or `Stdlib`'s spelling)
  written directly into a
  native claim (`Verdict.p`, `pf`, `claim`, `claimf`, `pass_fail`, `gated`, opened or qualified) and for one
  reached through a file-local binding, helper, wrapper or module (gh-ocannl-801, gh-ocannl-887,
  gh-ocannl-908). The reader is `test/support/verdict_provenance.ml` (gh-ocannl-931): ONE walker
  over the syntax that models each form once as a scope-and-polarity *provenance* -- two views,
  what the value's `true` can rest on and what its `false` can, each listing the quantifiers (or
  not-yet-known function parameters) the value may reduce to and the populations certainly
  non-empty at that polarity; a constant when the value is one; a closure when it is a function.
  Every question the old walkers answered separately (what a binding returns, what it depends on,
  which parameter a wrapper claims, which function a call resolves to) is a projection of that one
  model, so a syntax form handled for one question is handled for all: `not` swaps the views, `&&`
  joins them (sources union, witnesses union, and a witness covers a source the moment they meet),
  a conditional merges its branches (a condition steers between values and IS the value only where
  a branch returns a constant; alternatives all returning one constant fold to it), `|>` and `@@`
  are rewritten to plain application, and every binder -- `let`, parameters with their optional
  defaults walked before the parameter shadows anything, match cases mapped to the scrutinee's
  components, `open`, `module` -- goes through one environment. Population identity is lexical (a
  name plus its binder), which is what stops a witness from vouching for a shadowed collection.
  Functions are closures: parameters are bindings whose value is the parameter itself, and applying
  a closure substitutes the actuals -- a Boolean parameter by the argument's provenance, a
  population parameter by the argument's identity (so a caller's guard covers a helper's quantifier
  over the same actual and no other), a destructured parameter by its component of a literal tuple
  or record. `Verdict.p` itself is such a closure with a label slot and a value slot, which is why
  a local wrapper, a partial application of a native claim, a `function`-case wrapper and a claim
  inside a callback are one mechanism. A claim fires when the closure receives its last required
  argument; the diagnostic names the innermost binding that wrote the quantifier (a local of a
  function body is taken over by the function when it is referenced; a local nothing named
  encloses stays the name), or, for a quantifier written directly into a native claim, the claim's
  label -- which is also the exemption key for the sites where emptiness is the passing meaning
  and nothing in scope must be non-empty. The exact, stale-checked exemption list holds those and
  the bindings whose passing meaning allows an empty population; an entry names ONE definition,
  checked by definition offset rather than assumed, so a name shadowed by a second definition
  refuses the run instead of covering the body nobody read. Synthetic controls include two child
  processes the shipping ratchet demonstrably refuses (one per diagnostic), the mutation-run
  manifest `test/operations/verdict_ratchet_controls.md` (held equal to the control labels in both
  directions), and a generated syntax coverage matrix: every value form the model handles, under
  every quantifier kind and every spelling of the quantifier's function (qualified, `open List`, a
  module alias, a local open), in refusal / inverted / guarded / shadowed cases, printed as a grid
  in the golden so a missing cross-product is a visible cell rather than a review finding. Adding
  a syntax form means one case in `walk` and one family in `matrix_families`; the grid says whether
  the form composes with everything else.
  The literal- and computed-label exemption lists carry the same one-key/one-site contract
  (gh-ocannl-891). `Verdict_scan.site.position` is the parser's absolute character offset, while
  line and column are report text; `record_definition` aggregates all three exemption families by
  offset, and each refuses a key that resolves to more than one site. Multiline and same-line
  duplicate controls exist for both literal and computed labels, because a line number merges the
  exact same-line shape the check must distinguish. Direct quantifiers reaching a claim through a
  wrapper or a native call use the claimed ARGUMENT offset, so one intentional exemption cannot
  silently cover another call through the same wrapper or another claimed slot in the same call.
- `verdict_ratchet`'s third reader is about DIALECT (gh-ocannl-997): `Verdict_scan.dialect_census`
  refuses a label reported through both `pass_fail`/`pass_fail_all2` and `skipped` in one source,
  since a skip prints `p`'s line and the pair breaks the golden on the host that skips. The rule,
  the remedy (`Verdict.gated`) and what the reader cannot see are in the conventions note. A new
  claim entry point that takes a label and a boolean joins `claim_kinds` in
  `verdict_provenance.ml`, or the quantifier reader goes blind at its call sites; a new
  `Verdict.Claims` member of any kind joins `Verdict_callee.members`, which `verdict_ratchet` holds
  equal to the module. The refusal
  reader cannot go blind that way: `refusal_control_scan_cases` holds `refusing_claims` plus
  `non_refusing_claims` (`refusal_control_scan.ml`) equal to the members `Verdict.Claims` binds,
  read from `verdict.ml`, so a new `Claims` member is red until judged in one of them.
- Whose a claim is -- `Verdict`'s or not -- is ONE resolver's answer (gh-ocannl-1361):
  `test/support/verdict_callee.ml` walks a source once through `Lexical_scope` and every Verdict
  reader (the provenance layer, `Verdict_scan.claim_calls` behind the line-break and dialect
  readers, `gate_census`) looks the callee up by its location. Before it, each reader decided for
  itself and they drifted: staging#1150 fixed the line-break reader in six rounds, one edge per
  round, while the provenance layer still read a `module Verdict = Local_printer` as `Verdict`. A
  claim name that reaches `Verdict` only past an `open` of a module the resolver cannot read (any
  but `Verdict`, the source's own structures and the standard libraries) is `Ambiguous`, whether
  unqualified or qualified (`Verdict.p` may be that module's `Verdict`): still read, and its
  refusal adds the qualify-the-call remedy. The resolver answers LEXICALLY; the provenance layer
  alone carries a module through a functor application (`Make (Verdict)`), asking the resolver only
  whether the argument's path denotes `Verdict`. A question about Verdict ownership goes into the
  resolver, never into one reader, and `verdict_ratchet`'s agreement controls -- one claim each
  reader would refuse, spelled every way the resolver distinguishes -- fail when the readers
  disagree.
- `verdict_ratchet`'s fourth reader is about GATES (gh-ocannl-1289): `Verdict_scan.gate_census`
  refuses a `Verdict` `gated_all`/`gated_alli`/`gated_exists` claim whose `~when_` reads the
  population it quantifies or a collection that population is built from. Such a gate closes when
  a regression empties or changes the population, so the failure reads as a skip — the
  gh-ocannl-1115 trap, shipped twice in wave 2026-10-05 (a gate on the claim's own `results` being
  non-empty; one on `List.exists real` over `results` mapped from `real`). Both sides are compared
  by lexical identity, a name plus its binding. The GATE side is everything `~when_` reads, through
  every `let` and every write it reaches. The POPULATION side is what the quantified argument is
  BUILT from — `let` right-hand sides, branches and cases but never a condition or scrutinee, data
  thunks, `!`/`ref`/`@`, writes into a mutable container (`x := e`, `Queue.enqueue x e`), and the
  collection-carrying arguments of a standard collection module's functions, resolved through
  aliases, shadowing and lexical `open Base`/`open Stdlib` — but never another call's arguments.
  That asymmetry is the rule: a capability gate may read the program, limits and backend the
  population is computed FROM, never the population. The reader's header in
  `test/support/verdict_scan.ml` states the exact forms, which arguments of which library
  functions count (selectors, generators, Stdlib order), and the boundary: a population threaded
  through a function parameter, a record field, an unqualified call under `open List` or a
  destructured `let` is not read; a gated application the reader cannot complete is refused as
  unread rather than counted clean. `Verdict.gated`'s boolean is read too (gh-ocannl-1361): its
  population is what the boolean's quantifiers (`for_all`, `exists`, `is_empty`, ... of a
  collection module) range over, through names, writes, `not`/`&&`/`||` and branches, so
  `gated ~when_:(not (List.is_empty xs)) label ((not (List.is_empty xs)) && List.for_all xs …)` is
  refused; a boolean that quantifies nothing has no population. Its recognized forms are held
  equal to the `gated` and `gated_*` members of `Verdict.Claims`. A gate folded into the claim's value
  (`p "…" ((not on_cpu) || ok)`) is the fifth reader's (below). LEG claims are gated claims to it
  (gh-ocannl-1794): an `in_leg`/`in_legf`'s gate is the `~when_` of the qualified `Verdict.leg`/
  `Verdict.sub_leg` its `~leg` reaches -- through names, every `sub_leg`'s outer leg, an `if`'s or
  `match`'s pick of a leg (its condition, scrutinee or guard a gate too) and a local function's body --
  and its population is what the claims its make function makes range over, and, on a path where
  none of those runs, everything the make function hands a helper. Both sides follow data thunks (`lazy`,
  `Lazy.force`), the population side for `gated_*` claims too. A leg handed in as a parameter or
  read from a field is the boundary, counted on stderr as uncompared, as a `gated_*` population
  through a parameter is; a leg claim applied in part is refused as unread. Most of its mutation runs exit 4, not 1: the ratchet's trailing
  `OK:` row prints only on success (the manifest's gh-ocannl-1289 section says how they were
  read).
- `verdict_ratchet`'s fifth reader is about a gate FOLDED INTO THE VALUE (gh-ocannl-1310):
  `p "…" ((not on_cpu) || ok)` prints `true` on a GPU, a pass byte-identical to a CPU run that
  checked `ok`, so the transcript says the claim held on hardware that never evaluated it.
  `Verdict_scan.folded_gate_census` splits each `Verdict` claim's value (the boolean, or a
  quantified claim's `~f` body; `p_none`'s at the opposite polarity) into the disjuncts that alone
  pass it -- through `||`, `not`, a negated `&&`, a `let`-bound name, and an `if` whose other branch
  is the passing constant -- and refuses one that is RUN-CONSTANT beside one the test computed. Run-constant
  is what a configuration reader returns (`get_global_arg`, `getenv`, `Context.backend_name`,
  `Context.hardware_limits`, `Sys.win32`, `Utils.settings`), and what library calls compute from those and constants
  alone, through names anywhere in the source: `on_cpu`, `seeds_here`, a `lazy` hardware probe. A
  value that also reads a parameter, a pattern variable or a computation (`Context.auto ()`) is the
  test's own, so an implication over a computed antecedent (`(not ran) || finite`) passes, and so
  does a census computed under the backend. The remedy is `Verdict.gated ~when_` (a `gated_*` form
  for a quantified claim; a branch with `skipped` for a pairwise one, which has no gated form). Its
  header in `test/support/verdict_scan.ml` states the boundary: claims through a wrapper or a
  pipeline, a source function's result, and a fold below a conjunction (`ok && ((not on_metal) ||
  dialect)`, partly checked everywhere) are not read. The tree's 32 folded sites converted with
  their goldens unmoved, `gated` printing `p`'s line; the cross-machine sweep's skip coverage now
  sees what they never checked.
- Boolean operator aliases in `verdict_provenance.ml` (gh-ocannl-968) keep their first
  argument until application, so constant annihilators and partial applications use the same
  conjunction/disjunction algebra as direct syntax. Partial Boolean predicates retain their captured
  operand views, and unresolved calls forwarded through helpers move to the enclosing closure. Implicit aliases after an unknown module
  open are opaque; known qualified primitives and local definitions remain resolvable. Boolean
  ordering uses that algebra after length witnesses only for known standard operators and Boolean
  operands. Direct parameter returns transfer their Boolean identity, constants, and callable closures; ordering over
  unresolved parameters reuses deferred-call placeholders until substitution establishes the operand
  types. Optional defaults retain their result metadata until supplied or omitted arguments resolve
  it. Aggregate payloads remain distinct from Boolean results. A deferred call carries its
  argument's lexical population key alongside the walked value; only an exact whole-argument parameter inherits that key. Curried functors retain
  each residual module expression with its captured environment, binding one argument at a time.
- One-shot negative controls use `tools/mutant.sh <module> <perl-expression> <@alias>`
  (gh-ocannl-1880). It refuses a byte-identical edit, requires a green unmutated baseline,
  runs one test through `tools/test-run.sh` and reports false claims. Exit 1 means caught;
  incomplete or never-executed runs exit 4, and a red baseline refuses with 2. It shares the
  restoration and cancellation machinery below.
  For manifest evidence and reusable literal patches, mutation manifest rows use `tools/mutation-run.sh <module> <patch-file> <@alias>`
  (gh-ocannl-969), in an otherwise idle, isolated worktree. The patch is literal bytes
  `OLD@@@NEW`, with exactly one delimiter, a nonempty OLD occurring exactly once (overlapping
  occurrences count), and no implicit newline trimming; NEW can be empty. Where OLD occurs nowhere
  byte for byte it may match modulo whitespace (gh-ocannl-1429), so an ocamlformat rewrap of the
  target -- a long string broken at a space with a `\`-newline continuation, a tuple's parenthesis
  padded once it no longer fits a line -- does not strand a patch written before it, nor the
  reverse: a whitespace run of OLD matches any run, whitespace just inside a bracket is optional,
  the match must still be the only one, NEW replaces the matched bytes verbatim, and the report
  says `anchor: matched modulo whitespace`. A rewrap that moves a token is still a rewrite.
  `--batch <module> <patch-dir> <@alias>` runs every `*.patch` of a directory against one module
  and alias: all patches are anchored before anything runs, the baseline runs once, each mutant
  runs and is restored in turn under a `== <patch> ==` header, and a table ends the batch
  (`caught`, `SURVIVED`, `NO EVIDENCE`, `NOT RUN` after a signal); it exits 0 only when every
  mutant was caught, 1 when one survived, 4 when one was no evidence. The runner first runs
  the unmutated test once (`baseline: <run>`) and refuses, with nothing mutated, when that
  baseline is red: a golden not yet promoted after a change to the test read mutants as stopped
  early (gh-ocannl-1361's worker, twice), and a test red on its own would read every mutant as
  caught (gh-ocannl-1364). That is one more test run per mutation. It then invokes
  `tools/test-run.sh` on the mutant, prints its actual verdict and run id, and extracts every complete
  `FAIL: ...: false` line from that run's full log, and every failed `pass_fail`'s
  `FAIL: ...: FAIL` or `FAIL: ...: FAIL (<detail>)`, preserving order and duplicates -- apart from
  `Verdict.fail` messages, whose line the stdout carries too, listed under `failure messages`
  (gh-ocannl-1364). The alias
  names ONE test (`@<dir>/runtest-<name>`, `slow-<name>`, `train-<name>`) with a golden
  `<dir>/<name>.expected`, refused otherwise, because 1 means "caught" and 0 "survived" only for a
  run that reached its last row (gh-ocannl-1083). The runner snapshots the test's stdout
  candidates (`<name>.exe.output`, `.actual`, `.output` under `${DUNE_BUILD_DIR:-_build}/default`,
  where dune leaves them even when the test exits nonzero) before launch, counts the rows of the
  one this run rewrote against the golden's (`rows: N printed of M`; its own `FAIL: ` rows beyond the
  golden's are no rows of it, `rows not counted: K`, gh-ocannl-1364), and exits 4 in place of 0 or
  1 when the run is no evidence: `STOPPED EARLY` for fewer rows or Verdict's own `STOPPED EARLY`
  line in the log, `NEVER RAN` when no candidate changed (a build failure, or a mutation that left
  the executable byte-identical so dune reused the old result -- a previous run's output is never
  counted), `NOT COUNTED` when two did. A raising `Verdict.case` prints one claim in place of its
  rows while the cases after it still run, so fewer rows still count as reached (`cases raised:
  K`, gh-ocannl-1084) when a case raised, Verdict's teardown line (`FAILED: n checks …`) shows the
  process ended through `exit` rather than a signal, no `STOPPED EARLY` says that exit came from
  inside a later case, and the stdout ends on the golden's last row or on a case's raise. Rows
  short only by refusal-manifest marker rows (`  [scanner-refusal:<md5>] …`) count as reached too
  (`refusal markers omitted: K`, gh-ocannl-1216): a claim that failed prints no marker, so a
  complete failing run of a manifest-printing scan is a row short, and it counts when every other
  row is the golden's, row for row (a claim of it may read false; the run's own `FAIL:` rows are
  skipped -- `Verdict.fail` prefixes EVERY line of a multi-line message, so an "expected/got"
  continuation is skipped too rather than read as a row the golden lacks), and Verdict's teardown ended the process; a run missing or swapping a row, or with no
  teardown, still exits 4. Each omitted marker is listed with why it did not print (gh-ocannl-1499):
  the failed claim behind it, as `print_section` named it on stderr in a `Refusal marker not
  printed (not part of the golden): <marker> -- <reason>` line (`Refusal_section_heads.not_printed_head`) -- `its claim
  failed: "<label>"`, or for a direct failure `the claim answering for it failed: "<label>"`, its
  `raw_direct_evidence` claim (`the claims answering for it failed: "<a>", "<b>"` when several of
  its entries' claims failed) -- or, with no such line, `no failed check of this run is behind it:
  its claim or refusal did not run` (a reworded diagnostic gets `its diagnostic is no longer in the
  source`). The printer names only failures: a run prints a subset of its row (the rest belong to
  another control golden), so "did not run" is every passing run's normal state, and writing it
  would put hundreds of lines on every green scan's stderr. And the printer decides, not the
  runner: only it holds the format a claim's label must match and the evidence table, where a
  marker's fragment would only guess; `why_not_printed` takes the table as an argument and
  `refusal_control_scan_cases` pins each arm. A golden's success trailer is read the same way (`success trailer
  omitted: K rows`): the scans print their `OK: ` rows through `Verdict.success_trailer`, which
  prints them only while no check has failed, most of them behind a blank row of the trailer's own
  (`~blank_line:true`), so every complete
  failing run of `verdict_ratchet`, `env_var_deps`, `config_dep_completeness` and the rest is
  that many rows short; the runner matches the run against the golden without its `OK: ` rows,
  with or without the blank row before them, under the same row-for-row and teardown conditions.
  A new scan that prints its trailer through `Verdict.success_trailer` inherits this. A golden
  ending on a marker section's closing row (`End of synthetic controls.`, every manifest golden)
  needs neither the teardown nor the row-for-row match: a run whose stdout ends on the golden's
  LAST closing row (counted, since every section ends on the same text) printed all it was going
  to, so its shortfall is omissions, never a cut (`rows short of the golden: K`) -- a mutant that
  silences one refusal while every claim passes, with no failed check and so no teardown, is
  caught -- and a run cut inside the markers lacks the row and stopped early (gh-ocannl-1251,
  gh-ocannl-1318). Elsewhere the teardown is still all that separates a complete failing run from
  one cut before its last rows. A test that echoes a child's stderr (`verdict_teardown`)
  can carry a child's `STOPPED EARLY` line; read the rows line before discarding such a run.
  Otherwise the exit status remains test-run's, so a killed mutant normally exits 1, and a
  passing mutation exits 0 and needs investigation.
  INT, TERM and HUP cancel and reap the run before restoring; restoration is confirmed by `cmp`
  against the backup, including CRLF and a missing final newline. `tools/test-run.sh idle`
  probes its existing worktree flock (0 idle, 3 held, 2 unreadable); this is a point-in-time
  snapshot, not a reservation. A busy preflight refuses before mutation. If a killed launcher
  or supervisor leaves a lock holder alive, restoration is deferred: inspect/stop the worktree
  runs before recovering the source manually. Refusal exits 2; deferred or failed restoration exits 3 and retains the printed recovery copy; after SIGKILL, use that copy
  manually. It is temporary storage, without a power-loss or reboot recovery guarantee. `tools/test-mutation-run.sh` drives the shipping
  runner and test-run supervisor in isolated fixtures; Ubuntu's shell-harness CI runs it. The
  Verdict text the runner reads it takes from `verdict_teardown.expected`, whose last section is
  its children's streams, and replays through the runner: a refactor of `verdict.ml` keeps the
  harness green, while a promotion that changes those transcripts means running the harness.
  The same holds for what the runner skips of a scan's ending: the `manifest_held`/`manifest_failed`/`manifest_silenced` trio ends
  through the printers every scan shares, `Verdict.success_trailer` and
  `Refusal_control_manifest.print_section`, and the failing child's run, replayed against the
  passing child's stdout, must read as caught with one marker and the trailer omitted, and the
  silenced child's run, with no teardown, as caught with one marker omitted -- no source grep of
  either printer, nor of a scan (gh-ocannl-1364). The replay also reads each omitted marker's
  reason: the failing child's names a claim the runner listed as false, and the silenced child's,
  named nothing, reads as did not run (gh-ocannl-1499).
- The guarded pairwise claim has the same two label dialects as the scalar claim (gh-ocannl-816):
  `pf_all2` formats a computed label before taking the two arrays, and `pass_fail_all2` preserves
  `pass_fail`'s lazy failure detail while adding the structural empty, floor-shortfall, or
  length-mismatch reason. A compound claim that joins independent parity pairs stays out of a
  many-pair combinator: split it into separately labeled `p_all2` claims, so the transcript names
  which readback failed rather than merely saying that one of them did. Pairwise distinctness has its own
  combinator, `p_pairwise_distinct name xs ~equal ~to_string`: it refuses fewer than two source
  values, since distinctness over one element is as vacuous as `for_all` over none, and reports the
  first collision by `to_string` on stderr.
- A float readback compared under a tolerance goes through `p_close name got want ~tol` (and
  `pf_close`, `pass_fail_close`; gh-ocannl-1224), `tol` answering the admitted difference for a
  reference value (`fun _ -> 1e-3`, `fun w -> 1e-3 +. (1e-5 *. Float.abs w)`), never `p_all2` with
  a hand-rolled `Float.(abs (a - b) < …)` predicate. Its value failure keeps `p`'s bare `false`
  line and adds, on stderr only, the differing-cell count, the first and worst cell with both
  values and the tolerance there, max |want| and `?what_ran` -- what gh-ocannl-1217 needed a
  remote reproduction round to learn (5 ulps at magnitude 2e5 in every seed: reference rounding,
  not a seed). `p_all2`'s own value failure adds a count and first index there; neither detail
  line carries the `FAIL: ` prefix `tools/mutation-run.sh` counts. An aggregate claim over
  candidates whose count no golden pins (`p_all "every seed matches" runs ~f:…`) takes
  `Verdict.within` as its predicate: `p_close`'s check and stderr line, no claim of its own
  (`schedule_strided_1x1` is the exemplar), and so does a conjunction or a `gated` value that must
  stay one golden row (`p name (Verdict.within name a b ~tol && Verdict.within name c d ~tol)`).
  `verdict_ratchet` refuses the hand-rolled form (`Verdict_scan.tolerance_census`): a parity
  predicate applying `abs` to a difference, inline or through a name bound to such a function,
  partial applications included; and, in any other claim's arguments, a function BOUNDING an
  abs-of-difference (`<`/`<=`) or a name bound to anything holding one -- a helper, or a value
  computed through it (`let ok = Array.for_all2_exn got want ~f:approx in p "x" ok`; gh-ocannl-1470).
  An unbounded error helper or abs-of-difference reduction (a fold, `fold_left`/`fold_right`,
  `max_elt` or `reduce`, applied or piped) is tracked through names and arithmetic until an upper
  comparison bounds it (gh-ocannl-1837), and so is a `ref` a loop or callback writes an error into
  (`max_err := Float.max !max_err (Float.abs (a -. b))`, by lexical binding, renamings included;
  gh-ocannl-2122). A scalar bound written in the claim itself or on one cell's error, and a lower
  discrimination margin (`abs (a - b) > m`) stay unread by design. Ordering two error metrics is a
  scalar claim too. Array-cell, field and parameter writes, a mapped error array bounded
  element-wise, and a shared helper of another module are beyond the reader (its header in
  `test/support/verdict_scan.ml` lists them).
  `Ll_test.close` and `Ll_test.same` use `Verdict.within` directly,
  retaining their empty-array and subtraction-only semantics. Migrations of legacy error folds use
  `~difference_only:true` as compatibility with the original subtraction-based predicate,
  including its treatment of nonfinite values; a strict positive bound becomes
  `Stdlib.Float.pred bound`, and relative-error checks retain their original division to avoid
  changing boundary rounding.
- Guarantees that fire only on an empty collection are never exercised by a green suite, so
  `verdict_quantified` stages them: the satisfied forms run directly, and each refusal runs as a
  CHILD process whose streams the parent captures. Capturing is not tidiness — a refusal prints
  `FAIL:`/`FAILED:`, this repository's failure marker, so an inherited stream puts those words in a
  green run's log and costs `grep FAIL` its meaning (the same argument `generated_provenance`
  makes). The second half of that test is the one that makes a wide sweep safe: it runs `p` and
  `p_all` in two children and requires their stdout to be equal, which is the property "converting
  a site does not move its golden" stated as a check rather than as a hope.
- Executed parity needs a NON-DEGENERATE operating point (gh-ocannl-1135): the raw `0..223`
  ramp in `test/operations/online_softmax.ml` saturated attention into a one-hot softmax, leaving
  score-path parameter gradients of order `1e-19`; agreement then said little about the backward.
  Its `model` now scales the ramp into `[0, 1)`. Leg 6 prints each parameter's reference scale
  (`max abs` of the composed gradient) beside the worst normalized difference to stderr, tagged
  `(not part of the golden)`. Read that scale against the ACTUAL comparison: `close` accepts
  `abs(got - reference) <= tol * max(1, abs(reference))`, so below magnitude one the absolute
  floor is `tol`, not a purely relative bound. A reference at or below that floor can agree with
  zero and prove no meaningful parity; merely asserting it is nonzero does not fix this. Choose
  inputs that exercise the intended values and gradients above the floor, and report their scales
  alongside the errors; retain saturated or zero cases as separate edge-case coverage. Leg 6's
  scale report is diagnostic, not an enforced lower-bound claim.
- Executed parity cannot say a routine RAN tensorized (gh-ocannl-1283): a `Tile_mma` that declines
  at codegen renders the lane-0 scalar fallback, which computes the same values, so only the
  routine's mma census (`Context.routine.mma`, `Ll_test.execute_with_census`) tells the two apart.
  `tensorized_census_ratchet` (gh-ocannl-1309) holds it: a `Verdict` claim whose label names
  tensorization (`tensoriz`, the word `mma`, ...) AND asserts parity (`match…`, `bitwise`,
  `parity`, `reference`, ...), in a top-level item that executes a routine, is refused unless that
  item -- or a top-level helper of the file it references -- JUDGES the census
  (`Ll_test.rendered_tensorized`/`all_tensorized`/`census_agrees_with_capability`/..., a
  `C_syntax` tensorization comparison, constructor or field). Binding the census and discarding it
  does not count, and a claim in a check helper is also each executing caller's, which must judge
  too. The closed lists and the boundary are in `Test_scan.Tensorized_census_scan`'s
  header; the two that matter when writing a test: the granularity is the top-level item, so one
  judgment covers every leg in a big `let () =`, and a label whose tensorization word arrives
  through a function parameter is not read -- name the claim's tensorized leg in its literal text
  and judge its census beside it.

## Hand-built IR tests

- To read the IR a hand-built test optimized, run it with
  `OCANNL_OUTPUT_DEBUG_FILES_IN_BUILD_DIRECTORY=true`: `Ll_test.optimize` writes
  `<name>-unoptimized.ll` and `<name>.ll` under the gate a backend's own lowering uses, in
  `_build/default/<test dir>/build_files/<exe>/` (`ll_test_dump` pins it). `optimize_scoped` optimizes
  `raw`, so those two files are `raw`'s; the backend's source for the routine shows the scoped program.
- `ll_test_ratchet` prints an adopted-but-still-hand-building census (gh-ocannl-1090): the same
  syntactic record and private-traversal counts, sorted by source path, for adopted sources with
  either count nonzero. These golden rows are informative and promotable, with no growth cap;
  unadopted migration debt retains its independent refusing caps and stale-exemption checks.
- `Ll_test`'s traversal is the one place a new `Ir.Low_level` constructor is handled, and it now
  carries the queries the hand-built-IR tests used to write for themselves. `walk` takes a record of
  hooks: the construct-specific ones, a generic `?on_stmt`/`?on_scalar` for a counter that names its
  own shape, and an `?on_exit` — which is what makes an enclosing-context query derivable from the
  same walk instead of a second one. `?in_scopes:false` selects the statement-positions-only reading
  (the walk `Schedule.find_loop` had before gh-ocannl-668); every other query leaves the
  `Local_scope` descent on, and deciding it here once is the point, since `schedule_partition` alone
  had grown six walks each deciding it independently while scope nesting was the property under
  test. Above them sit `loop_sites`, `find_loop`, `find_loop_with_extent`, `find_nest`, `binds_loop`,
  `count_loops`, `first_binding` and `census`. Reach for `find_nest ~outer_n ~inner_n` rather than
  locating by extent alone wherever an initialization loop can share the reduction nest's extent:
  matching on extent takes the earlier one in preorder, which is how a `Partition` leg came to unroll
  an init loop and report `copies = 1` while exercising nothing. The walk also descends into
  `Tile_mma`'s `fallback` rather than stopping at the tile, and reports the operands through the
  fallback alone — reporting the tile's own `d`/`a`/`b` as well would count every operand of a
  tensorized nest twice.
- `Ll_test.nest_paths` is the other loop query, and not derived from the walk: the perfect-nest
  chain of each top-level statement (a loop, then the loop that is its body's only statement,
  through `If` guards), which a schedule test selects by length to name the i/j/k symbols its
  `Tile`/`Privatize` schedule addresses. Twenty-two tests carried a private copy until gh-ocannl-1091;
  `bin/schedule_bench.ml` and `bin/narrow_gebp_bench.ml` still do (benchmark binaries link no
  test library). It stops at anything other than `For_loop`/`If`, so a new constructor ends a chain
  rather than needing a case here.
- A recognizer that must know a node by IDENTITY gets its negative control end to end, through
  `Ll_test.substitute ~from ~to_` (every position a node has: accesses, `Tile_mma` operands, scope
  ids) and `Ll_test.look_alike ~id tn` (a fresh node with `tn`'s namespace, label, precision and
  shape). `substitute_optimized` plants the look-alike in a lowered or scheduled program, carrying
  its traced-store entry and placement over and leaving the identity sets naming the real node, so
  the forged program differs by identity alone; the transform that consults the recognizer must then
  decline. Probing the classifier on the look-alike tests only the predicate. Pair it with the
  positive arm (the look-alike enrolled, or the substitution reversed back to the captured
  program): `schedule_epilogue_fusion` and `online_softmax_block_mma` are the two shapes, and both
  went red under their pre-fix label rules. `substitute` is the second exhaustive map over the IR
  beside the walk, with no wildcard, so a new constructor fails to compile there too.
- The dynamic-indexing pair has builders of its own: `Ll_test.gather` (`Get_dynamic`),
  `Ll_test.scatter` (`Set_dynamic`) and `Ll_test.scatter_add`, the read-modify-write shape
  `rewrite_one_hot_reductions` actually mints. Reach for them rather than spelling the record:
  the `idcs` array must carry a `Fixed_idx 0` PLACEHOLDER at `dyn_axis` — a convention stated
  only in `low_level.ml`'s type declaration, and one a hand-spelled site gets to state again for
  every axis it happens to have — so the builders plant it and refuse a `dyn_axis` outside the
  array. Neither constructor reaches `optimize` through the ordinary pipeline (lowering emits
  neither, and the rewrite runs after both virtualization arms), so hand-built IR is the only way
  to put one in front of the analyses; `gather_table_placement.ml` is what that looks like.
- `Ll_test.interpret` evaluates a `Low_level.t` in-process, for a test running far more programs
  than it could compile: a compiled routine costs on the order of 100 ms, an `optimize` tens of
  microseconds (gh-ocannl-1964). It raises `Interp_fault` on what a compiled run would swallow: an
  access outside the node, a local read before any assignment, an unwritten cell of a node with no
  entries. Values are f64, so a program keeps them exact in its storage precision. It runs `Where`
  lazily, and a projection evaluates only the operand codegen renders. Codegen hoists a scope's body
  ahead of the statement, so a scope in a discarded arm runs on a backend though the lazy run skips
  it: `~strict:true` probes those scopes' addresses (values discarded, no node written, locals
  restored) and faults on one outside its node (gh-ocannl-2077; `virtual_inliner_oracle`'s
  `strict_probe_controls` pin the probe's boundary, and both its arms evaluate under it). It does not model `pp_local_defs`'
  emission order or a blending backend's eager arms. It is a test
  instrument: a test leaning on it pins it to a backend and to a reference outside it, as
  `virtual_inliner_oracle`'s Tier B and replay do.
- An executed reference must DISCRIMINATE, not merely run: give every producer a value that varies
  with every symbol of its iteration and stays clear of the init/sentinel value (`1 + i`,
  `1 + 10*outer + inner` — the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`). A
  constant producer replays an identical assignment under a too-wide range guard, a value omitting
  one symbol is constant along that axis under a wrong substitution, and a value colliding with the
  zero-init hides a dropped first iteration — each is a leg that passes for the wrong reason.
- A test of a ZERO-related transform starts every checked run from a state the routine must
  overwrite (gh-ocannl-1625): init elision, zero seeds and `= {0}` declarations, `Zero_out`
  elision and sinking, zero expansion and folding, Privatize's dropped zero, `Split_reduce`'s
  partials zero. A fresh device buffer is already zero, so a zero that never runs computes the
  same values and the parity check passes. Prefill each zeroed node with per-cell values clear of 0
  and of what the routine leaves there (`Context.set_values` between compile and run, or the
  `~seed` of `Ll_test.run_linked`), and claim the prefill differs from the result. `run_linked`
  (so `Ll_test.run` and `execute` too) does this BY DEFAULT (gh-ocannl-1774): every node the
  routine writes before reading (`Ll_test.written_before_read`: `outputs` less `inputs`, with
  host-initialized nodes left out) that the seed leaves out and the context holds no value for
  starts from `Ll_test.prefill`. `~fresh_zero:[(tn, reason)]` opts a node out, and raises on an
  empty reason or a node the default would not reach. A value the passed context holds is kept,
  so a run-twice leg that runs on the context the first run returned still reads its results. A root's device buffer outlives the context value it came
  from, so a leg that needs a buffer no run has written links in a fresh `Context.auto ()` root. The
  default stops at `run_linked`, and a claim that the prefill differs from the result names its own
  seed. A test that drives `Context.run` itself still gets the fill below, which is weaker than a
  prefill: one value, and only on a first write, so a hand prefill stays right for a test OF a zero
  transform. A node the
  schedule mints (the partials) has no host handle: run the idempotent routine twice and read the
  second run. A `Local` or shared-memory node cannot be prefilled at all: put its zero after a write
  in the same routine (`zero_out_local_decl`'s hand-built leg). The prefill must reach the
  transformed arm; a reference that keeps its zeros is indifferent to it. A reference that is a
  plain compile shares every default-on zero transform (`gpu_elide_covered_zeros` on GPU,
  `buffer_aliasing`'s sink when both arms set it), so it is no oracle for that one: use a closed
  form or the gate-off compile. `default_prefill` is the closed form for the localized-accumulator
  zero seed: `seed_first_write`'s peel on a GPU backend, `try_localize_serial_reduce` on cc.
  `schedule_ops`'s matmul twins both take that seed, so they cannot catch a seed that reads the
  node. Prove the leg with `tools/mutation-run.sh` and a drop-one-zero
  patch; the inventory and its mutants are in lukstafi/ocannl-staging's gh-1625 PR body.
- `debug_fill_fresh_outputs`, on in `test/config` (gh-ocannl-1873), makes every executed test
  discriminate against a missing write: `Context.run` starts each `written_first` node whose buffer
  nothing wrote from `Ir.Ops.debug_fill_value`, NaN or for integer storage a `0xA5` byte pattern. So
  a red that names a NaN is a finding, not noise: either a write is missing, or the test reads a cell
  no writer covers and should seed it (`Ll_test.blank`) or stop reading it. Rerun with
  `OCANNL_DEBUG_FILL_FRESH_OUTPUTS=false` to see it was the fill. A test whose `lowered_transform`
  replaces the lowered code owes the zero it dropped, as a seed between compile and run
  (`hardware_warp_shuffle`'s `state_dropped_zeros`): the replacement leaves the node written first,
  so the run starts it from the fill. A zero dropped AFTER the analysis
  is what it catches: `written_first` is decided before scheduling, so a schedule or codegen drop
  leaves the node there, and so does a dropped init of a write that does not read its target (a
  diagonal, whose off-diagonal cells then read the fill). An accumulation left without its init reads
  its own target first, which takes the node out of `written_first` and, embedded, out of the
  required inputs: it lands in `read_first_undeclared` (gh-ocannl-1966), which the key fills too, so
  embedded state really starting from zero declares it (`{ x; zero_valid = true }`, or
  `Tnode.mark_zero_valid`), unless an `=+`, `=-` or `=||` opens it, which states that start itself
  (`Asgns.accumulates_from_zero`, gh-ocannl-2013); a library op written with `=+` where `=:+` was
  meant is therefore not this check's to find. Gradients are declared at creation, so a gradient that lost its zero
  before the analysis stays unfilled. That class is filled rather than refused because the raw read
  set is conservative: a read of a padded margin (`test_padding_reset`) can land there, correct,
  and any write covering a read lands over the fill first. A liveness-aliased buffer is filled by
  location, but only a range's first occupant reads the fill: a later one that lost a zero reads what
  its predecessor left, which only a closed form or a planner-off twin shows. Nor can the key reach
  what has no host-reachable buffer of its own: a `Local` node, a schedule-minted partial.
  `fresh_output_fill` pins the contract and holds `test/config` to it; the mutant tables are in
  lukstafi/ocannl-staging's gh-1873 and gh-1966 PR bodies.
- Host-side exactness premises share an index reader with initialization: wrap the value function
  in `Ll_test.operand ~b ~o ~i`, which checks rank and bounds against `b @ o @ i` before
  evaluating it (gh-ocannl-2176). This catches out-of-bounds permutations, not in-bounds ones:
  also compare the premise's host sums with the executed reference, as `conv_twin_packing`,
  `masked_persistent_fragment` and `schedule_coalesce` do. Hand-seeded readers check against the
  node's actual dimensions before flattening; a flat offset alone can alias another valid cell.
- A test operand minted from a FLATTENED offset stops discriminating at sizes that divide its
  modulus, and does it while still looking right: `(row * stride + col) mod p` collapses to
  `col mod p` whenever `p` divides the row stride, so every row becomes identical. That makes a
  whole class of bugs invisible — a transform substituting or repeating the wrong row, panel or
  K-block computes the correct output, which no whole-output check, checksum included, can see, so
  it is found by review rather than by a red test. `Ll_test.cycle` (multi-index) and
  `Ll_test.cycle_flat` (flat offset) compute exactly the values the hand-written idiom did and raise
  `Invalid_argument` when the modulus is blind to an axis of the `~dims` handed to them, which turns
  a latent trap into a loud failure the moment a size arms it. Converting a site is therefore free:
  `Float.of_int (i % 13) *. 0.25` is `~modulus:13 ~offset:0. ~stride:0.25`, and `(x *. s) -. c` is
  `~offset:(-. c /. s) ~stride:s`, so no golden moves. The care is in `~dims`, which must be the
  operand's real row-major shape read off its `NTDSL.init`/`TDSL.ndarray` call (`~batch_dims` then
  `~output_dims` then `~input_dims`), not the `Array.init` argument. When the guard fires, keep the
  modulus and pass `~radix:h` with `h` coprime to the MODULUS (not to the dims) and off
  `1 (mod modulus)`: the key becomes the multi-index read in base `h`, so no axis can cancel at any
  size while the value set — which an exactness argument may rest on — stays put
  (ahrefs/ocannl#1024) — `cycle` also refuses a radix whose key would reach a different set of
  residues than the row-major one (a short axis can leave one unreached), but the ORDER of the
  values moves, so an argument about partial sums is re-exhibited, not inherited;
  `h = 1 (mod modulus)` makes a square operand its own transpose. Changing
  the modulus instead changes how many values there are, which is a numeric re-derivation wherever
  the values are load-bearing. What this does not buy is
  aperiodicity: the values repeat with period `modulus`, so a shift by `modulus` is a symmetry, and
  where the blocking factors are searchable a packed panel can repeat under `k -> k + p` and hide a
  panel-substitution bug just as well; the recipe with no shift symmetry at any lag is
  `bin/narrow_gebp_bench.ml`'s `mix`. A formula that is NOT a flat index —
  `(i0 + i1 + 2*i2 + 3*i3) mod 7` — goes through `Ll_test.weighted`, which computes it unchanged
  and raises when a weight is a multiple of the modulus, the same blindness read off the weights.
- `Bench_checksum.mix`'s injectivity is about its full 24-bit values, not any selected bits
  (gh-ocannl-1118): its low seven bits can alias whole rows and columns. `residue` reduces the full value for odd
  moduli and draws even-modulus residues from bit 7 upwards, because an even remainder carries
  the raw value's parity. A bounded value set
  still needs row/column distinctness checked at the operand's actual extent; the binary
  32x64 control in `bench_checksum_discrimination` pairs that claim with raw parity's failure.
- `operand_key_ratchet` (gh-ocannl-1018) is what makes the guard above reach more than its
  callers: a test source minting a value from a remainder of a hand-rolled multi-axis key is
  refused unless the key goes through `Ll_test.cycle`, `cycle_flat` or `weighted`, or the site
  names an exact, stale-checked exemption with its reason (a `File` row covers the guard's own
  source). One detector reads both spellings on the parse tree — the multi-index
  `~f:(fun idcs -> … ((idcs.(0) * n) + idcs.(1)) % m …)`, following names `let`-bound in the body,
  and the flat `Array.init (r * c) ~f:(fun i -> … i % m …)` — because the sweep that preceded it
  missed a flat site four lines from a multi-index one it listed: to a person reading a file the
  two do not look like the same thing. What it deliberately does not read is stated in
  `Test_scan.Operand_key_scan`'s header: a remainder of one literal axis, `i % d` with `d` a
  trailing run of the length's factors (unflattening), a key computed outside the closure, and
  mixers other than a remainder or bit mask. Masks, including a shifted mix masked with
  `land 1`, follow the same axis reader; the trailing-factor exception applies only to remainders.
  A converted site keeps its values, so a golden moves only where
  the guard fires; the conversion found one blind term, `schedule_mma_matmul`'s tf32 perturbation
  at modulus 3 over a `k = 24` row, which took `~radix:2`. Converting links `ll_test`, which does
  NOT retire an `ll_test_ratchet` row: linking for an operand helper is not adoption
  (gh-ocannl-1052). `operand_key_scan_cases` puts each spelling beside its nearest legitimate
  text and drives the shipping scanner over a synthetic tree holding the blind fixture it must
  refuse.

## Codegen text inventory

- A test asserting on generated code must establish that the artifact it reads is the one THIS run
  emitted; `Test_utils.Generated` is how (gh-ocannl-655). `build_files/<exe>/<routine>.<ext>` is a
  side effect of a compile, not a value the test holds, and two things detach it from the compile it
  describes: `test/config/ocannl_config` keeps `clean_up_build_files_on_startup=false`, so an
  artifact outlives its run indefinitely, and a second compile under the same routine name overwrites
  it within a run. Either way the assertion outlives the kernel it asserts on — it keeps passing, and
  keeps counting as coverage, after that kernel stopped being emitted at all (folded to a constant,
  erased by precision inference, fissioned into a differently-named routine). `Generated.init
  ~backend_name`, called before the first compile, empties this executable's own subdirectory, so
  existence IS freshness — no mtime, no clock granularity. What licenses that sweep is narrower than
  "the directory is scoped": only the DEFAULT, executable-derived subdirectory is inherently
  process-private, since dune runs one process per executable -- unless a second stanza runs the
  SAME executable (a `-tf32` rerun, a mode argument). Those stanzas share the subdirectory and must
  share a dune `(locks <exe>)`, or one run's `init` sweeps the other's artifacts mid-run: on
  Windows the delete is refused while the kernels are loaded, and elsewhere it succeeds and the
  swept run fails to compile or read (`cc_kept_library`; `schedule_conv_gemm`, gh-ocannl-1720).
  `generated_lock_scan` holds that across the tree (gh-ocannl-1749): a program whose own modules
  call `init` and that two or more stanzas run (`Dune_stanza_scan.program_runners`, plus a
  `(test)` itself) needs one lock every runner names, a relative name resolved per directory as
  dune resolves it, or its program stanza carries `; generated-lock: none -- <why>` where each run
  provably writes elsewhere (`cc_march_census`: every piece and emit child `chdir`s into a
  directory of its own first, and one lock would serialize the pieces gh-ocannl-1502 split).
  Any configured `build_files_prefix`
  is refused outright — a second executable can be given the same prefix, so deleting there is
  unsafe, and without deletion a deterministic compile's re-emitted identical kernel is
  indistinguishable from a stale one (deletion is the only write signal that does not depend on
  timestamp granularity). Tests that assert on generated code therefore leave the prefix at its
  default. `Generated.read` fails through `Verdict` on a
  missing artifact instead of answering `None` — the arm that some call sites recorded as `false` and
  others forgot. `Generated.arm` deletes one routine's artifact before a candidate's compile, which
  is what a loop reusing a routine name needs in order to attribute what it reads; reading one
  routine twice across changed contents is otherwise reported as an unattributed overwrite. Corollary
  for a leg this backend cannot evaluate: gate it and report `Verdict.skipped` rather than letting it
  reach the read, because an absent artifact is a failure here by design.
  The dune side of that: `init` READS `build_files_prefix`, which the ambient record in
  `ocannl_config` carries like every other key, so nothing is declared for it (gh-ocannl-1425).
  `env_var_deps` still refuses the call in a library module, a `(select …)` arm included: it
  empties the calling process's artifact directory, so from a library it runs in every executable
  linking it (gh-ocannl-723). The other way round is closed by `codegen_text_inventory`, which refuses a test source that reads
  `Utils.build_files_dir`/`Utils.build_file` itself.
- A test asserting on a kernel's STATEMENTS -- not the prepended builtins, which spell every
  conversion and vector builtin, nor the signature, which names every node -- reads them through
  `Generated.main_logic` (text) or `Generated.main_logic_lines` (1-based line numbers, what the
  asm census's brace-range fallback takes), never as the text after `/* Main logic. */`
  (`C_syntax.main_logic_marker`). A large cc kernel is rendered outlined (gh-ocannl-1603): its main
  logic only calls its `ARRAYJIT_KERNEL_PART` functions, which come BEFORE it in the file, so the
  after-the-marker reading of an outlined kernel sees none of its statements -- a positive claim
  goes red and a negative one ("emits no vectorized store") passes vacuously. The helpers splice
  each part's body in at its call, finding the parts from the definitions the renderer emitted and
  the calls the main logic makes, and fail the run on a part no kernel calls (gh-ocannl-1678).
  Test-size kernels are never outlined at the default part weight, so check a reader under
  `OCANNL_CC_KERNEL_PART_WEIGHT=1 OCANNL_CC_COMPILE_UNIT_MIN_BYTES=0 OCANNL_CC_COMPILE_JOBS=4`;
  `test/operations/cc_kernel_outline` pins the splice against a whole compile. Two regular runs
  force it (gh-ocannl-1708): `test_uniform_virtual_lane-outlined` (the plain golden, flags on its
  rule) and the census's `emission-outlined` piece (its emit children are forced), each claiming
  the outlining itself through `Generated.outlined_parts` -- over the run or the main kernels,
  since a one-statement kernel stays whole at any weight. `branchless_select-outlined` (gh-ocannl-1778)
  joined them after the sweep's `cc-outlined` unit found its after-the-name reading blind to the
  parts. And `codegen_text_inventory` refuses a
  `substr_index` on the banner (`Codegen_text_scan.reads_after_marker`), exempting by name only
  `cc_kernel_outline`'s negative control.
- A test asserting on the ARRAYS a kernel declares reads them through `Generated.declared_arrays`
  (gh-ocannl-1778), never by scanning lines after `/* Local declarations and initialization. */`:
  each kernel's body ahead of its main logic, then the arrays its statements declare over
  `main_logic_lines`, stack and heap forms, each with its scope. Outlining
  splits them: the function-scope arrays stay in the kernel's frame (its parts take them as
  pointer parameters), while a block-scope one -- a pool `Grid` chunk's privatized tile -- moves into
  the part its loop went to (`Block_scope { part = Some _ }`). Two forced reruns pin the reading:
  `thread_private_gate-outlined` (the function-scope count against the estimate) and
  `heap_local_arrays-outlined`, whose Grid routines lead with a second statement so they outline
  at all, and which claims each Grid tile was read from a part.
- **Before changing code generation, read the inventory**: `dune build
  @test/operations/runtest-codegen_text_inventory` prints, as its golden, every file in the tree
  that pins the TEXT of emitted code (gh-ocannl-712). Two populations, and no single search finds
  both. Goldens holding emitted kernel or IR source live in `test/` and in `arrayjit/test/` — a
  scan of one tree is how gh-ocannl-623's first CI run went red, since three `arrayjit/test`
  goldens quote emitted constants. And some tests pin emitted text from a string literal in the
  `.ml` rather than from a golden, which no `.expected` scan can see; those are the expensive miss,
  because they are `Verdict` claims and so exit nonzero, failing a plain `dune build` rather than
  only `dune runtest`. Each source entry itemises the fragments it pins, `sprintf` formats and
  concatenations included with the hole shown (`"(float)(" ^ ... ^ ")"`, `"< (int)(%d.0))) {"`) —
  a range guard's bound is a float `Constant` at index precision, so gh-ocannl-623 turned
  `(int)(33)` into `(int)(33.0)`, a context nobody would think to grep for. Grep the inventory for
  the spelling you are moving, re-run what it names, and promote its own golden last.
- **It names the aliases too, read from the dune files** (gh-ocannl-1533): each member line ends
  with the aliases that re-run it, and the golden closes on their union, one per line, ready for
  `tools/test-run.sh run build`. A file is not yet a command — a test's own `runtest-<name>` diffs
  only its own `.expected`, while its kernel snapshots `<name>.<ext>.expected` are compared by the
  hand-written `runtest-<name>-extension` rule, so running a codegen test by its name leaves every
  snapshot unchecked (staging#1326). `Test_scan.Rerun_aliases` derives them: a golden from the
  rule whose `(diff …)` names it as the EXPECTED side, matching each `%{…}` against a substituted
  file name and printing the substitution (`when %{read:../config/ocannl_backend_extension.txt} is
  cu (cuda)` — that snapshot is checked only by a CUDA run), or from the `(test)` stanza beside it; a
  source from the stanza owning its module (`(modules …)` listed, else `:standard`, a `(select)`
  arm by its target), then the test's alias, the alias of each rule running its executable — or,
  for a rule that only writes targets, of the rules reading them — or an inline-test module's
  runner rule. A member none of that reaches fails the run rather than leaving the list short:
  give it a per-test alias. The backend a condition's value means is named by RUNNING the code
  the configuration reader writes the file with (`test/config/config_reads.ml`, gh-ocannl-1581)
  over every backend, never by a table beside the scan: the golden then splits the conditional
  aliases per backend, the list to run on that backend's box, and a value no build writes (a
  snapshot under an extension no backend emits, a log golden named after a deprecated alias) is
  refused as a golden no run compares; `marker_backend_vocabulary` holds the reader's extensions
  and alias resolution to `Backends`. Both lists stay apart from the kernel-text section's GPU list
  below. The derivation reads the rule shapes this repository spells; it is a
  point-in-time reading of the dune files, not of `dune rules`, and `tools/test-run.sh affected`
  stays the reach-by-linking answer.
- **The same golden then lists the tests that build their own kernel text** (gh-ocannl-1685): they
  pin nothing, so the text census cannot see them, yet a codegen change breaks them first. When
  gh-ocannl-1338 renamed every GPU node identifier to `t_<name>`, all 103 text pins passed on CUDA
  and HIP and master still went red: `shipped_kernels_alone`'s staged callback writes
  `sk_staged[0] = 2.0f;` verbatim, and nothing renames inside a `Staged_compilation`.
  `Test_scan.Kernel_exec_scan` names three routes: CONSTRUCTING a `Staged_compilation` (a pattern
  matching it builds nothing); passing `?prelowered` to `Context.compile`/`compile_outcome`, which
  compiles a hand-held lowering as written (`link_delta`'s `~prelowered:true` links nothing and is
  not the seam); and calling a support value that reaches either, directly or through other values.
  That third route is how `Ll_test.execute` and `Bench_harness.time_shipped_kernels` (each shipped
  kernel compiled alone) are found. The support values are derived, not listed: every module some
  `(library)` stanza owns, in the test roots and in `benchmarks/runners/ocannl` (whose sources and
  `dune` the rule also hands over), closed under reference and `include`. The golden prints that
  frontier. A test reaches it by a qualifier, an alias of one, or an `open`. The reading errs toward
  listing: a staged callback counts even in a test that only renders it, since its text is opaque by
  design and the scan does not guess whether a compiler sees it. The section closes on its own alias
  list, the one to hand to `tools/test-run.sh run build` on each GPU backend. This is only the
  kernel-execution set: the applicable golden routes in the earlier alias sections remain required
  on each GPU backend too. For example, `@test/operations/runtest-micrograd_demo_logging` appears
  in those earlier sections but not in the kernel-execution list. A member with no alias
  fails the run, as a text member does. `codegen_text_scan_cases` pins each route beside its near
  miss, including the negative control: a fixture whose staged callback writes a node by its bare
  label.
- **Then the tests that call a backend's compiler themselves** (gh-ocannl-1772): a change to a
  backend's builtins prelude, its compiler flags or how its compiler is resolved breaks them, and
  neither section above lists them. `test_shared_builtins` writes `Builtins_cc.source` and a probe
  of its own to a file and compiles it with `Cc_backend.compiler_command`; the `*_emulation` tests
  do the same with the Metal and CUDA builtins under cc. `Kernel_exec_scan.compiler_entries` names
  the entry points as spellings (cc's command and flags, `Nvrtc.compile_to_ptx`,
  `Hip_backend.Impl.hip_to_code`, Metal's `Library.on_device`), matched through aliases and opens,
  and the support values reaching one are found as above. The golden closes on an alias list per
  backend whose compiler is called, run on that backend's box. A spelling no handed-over source
  still uses fails the run; that claim dropped `Hiprtc.compile_to_code`, which only `arrayjit/lib`
  spells, on its first run. A NEW entry point is not found by itself: add its spelling. A test that
  compiles with a hard-coded `cc` and no entry is not found either.
- **Pins one file away, behind a `test/support` helper** (gh-ocannl-1772): gh-ocannl-1736 moved
  the residency markers into `Ll_test`, and after that the inventory showed
  `masked_persistent_fragment` -- pinning them through `Ll_test.resident_regions` -- as a member
  pinning nothing, with a complete itemisation. `Codegen_text_scan.helper_predicates` derives the
  helpers from the support modules: a structure-level value with a text test whose haystack
  derives from its own parameter, a nested closure included. Their fragments are itemised by the
  literal discipline plus a record field read off a parameter (every literal the module builds
  that field with), a nested function's parameter (what the helper passes there), and the
  module's other helpers they call. Whether a call pins generated text is decided at the call, by
  the CALLER's provenance: generated text is a pin; text the model cannot decide is listed as
  "haystack unvalidated" with its boundary; ordinary text is no use. So the scanners' own text
  predicates stay off the list. The golden lists each called helper with its fragments and
  callers, and each caller says which helpers it pins through. A value that only forwards to a
  helper in another module is not followed.
- **A test reaches generated text three ways, and the inventory tags which.** Through
  `Test_utils.Generated` (the freshness-checked artifact reader); by opening `build_files/` itself,
  which two tests predating that module still do; or **in memory**, calling an emitter and
  rendering the document it returns, or handing one the buffer to write into —
  `C_syntax.compile_proc`/`compile_main`, `Low_level.to_doc`/`to_doc_cstyle`,
  `Canonical_render.emit`. That third route touches no artifact at all, and modelling only the
  first two made an entire scan root look empty: the `arrayjit` tests cannot link `test_utils` (it
  is a `neural_nets_lib` library), so every one of them takes it.
- **The emitter frontier is derived, not listed** (gh-ocannl-748): `Emitter_frontier` reads the
  compiler libraries' COMPILED interfaces — the `.cmi` files the inventory's rule depends on — and
  calls a value an emitter when its result is a `PPrint.document` (through tuples and options
  alike), or it takes a `Buffer.t` to write into, AND it is given something of the libraries to
  render. Types rather than sources: a renderer without a return annotation or an `.mli` exposes
  its document only through an inferred type, and a source scan would have to be told about it —
  which is the hand-maintained frontier again. Add a renderer to a library and it is on the frontier
  the day it is exported; nothing to update. The frontier follows the compiled interface:
  an `.mli` hides implementation helpers, while a module without one exports its inferred
  signature, including internal renderers. `C_syntax`'s public surface is bounded by
  `arrayjit/lib/c_syntax.mli`.
  - Both types are followed through the library's own abbreviations for them (`type rendered =
    PPrint.document`), since an interface records the path a declaration spells rather than what it
    abbreviates. Those are keyed by the module path they are declared in, never by the bare name:
    `t` is declared in every module of every library, and a bare-name table made emitters of 400
    values returning some `t` or other (Codex round 1 on staging#487). A named module type is resolved the
    same way — `module M : S` exports S's values under `M`, which is the name a call site spells and
    an `open` brings into scope, so the value is attributed to both.
  - The second condition is what keeps generic names out. `Indexing.Doc_helpers.int : int ->
    PPrint.document` renders no program, and since the scan matches an emitter by NAME behind any
    qualifier, admitting it made members of every test calling `Bench_args.int` or
    `Random.State.int` — six files, three of them slow training tests. Those excluded values are
    PRINTED in the golden as `document combinators`, so a renderer that lands in that bucket
    (`vec_splat`, which assembles C text out of strings) is a line in a diff rather than an absence.
  - A derivation fails more quietly than a list: handed nothing, it reports a smaller census
    cheerfully. What the inventory pins is therefore a relationship — a library's wrapper interface
    DECLARES its modules (it is a list of aliases), and every declared module must be one whose own
    interface the run read. Both lists go to stderr. `emitter_frontier_cases` controls the rule on
    a fixture library whose `.mli` spells every shape and every near miss, and controls the
    tripwire by deriving from a wrapper alone in a directory of its own.
  - `arrayjit.utils` is named twice over, and needs to be: it has a module of its own name, so
    `utils.cmi` is that MODULE and dune's generated alias module — the one listing every member — is
    `utils__.cmi`. Reading only the former discovers the members `utils.ml` happens to alias, and the
    declared-versus-read tripwire goes on passing over the shorter list.
  - The rule asks for each library twice, and needs to: the object directory (`glob_files
    ../../arrayjit/lib/.ir.objs/byte/*.cmi`) is what exists in an ordinary build and what forces the
    interfaces to be built, while `%{lib:arrayjit.ir:ir.cmi}` is what resolves under `dune build -p
    neural_nets_lib` — each `.opam` file's build command — where arrayjit's stanzas are disabled and
    its libraries come from the switch. The member interfaces sit beside that wrapper in the
    installed directory, and `Emitter_frontier` looks for them there.
- **An indirection between the call site and the emitter is followed wherever one file can see
  it.** In the library, a transparent type alias (above). In the test, four shapes, all of which
  left the FILE listed while the fragment it asserts on went missing — the invisible-omission shape,
  since a member itemising nothing looks exactly like a member with nothing to itemise. A module
  alias (`module CR = …` then `CR.emit`) was always fine, since emitters match by name behind any
  qualifier; `Codegen_text_scan.emitter_aliases` adds the value alias (`let write = CR.emit`) and
  the wrapper (`let write ~buf p llc = CR.emit ~buf p llc`, whose own parameter is where the
  caller's buffer arrives — by label, or by position among the unlabelled arguments); and
  `scope_of` resolves an `open` of an aliased emitter module — including one a FUNCTOR
  produced (`module Syntax = Ir.C_syntax.C_syntax (…)`, which is how every backend and every codegen
  test reaches `compile_proc`), by taking the functor's own name — which the rejection below would
  otherwise miss. Membership, taint, the buffer destinations and the pin walk all go through
  the one resolver: rules that know different routes are how a file stays listed while its pin
  disappears.
- **Module aliases and literal `let`s resolve by lexical scope** (gh-ocannl-1079), over
  `Lexical_scope`, the model `ll_test_scan` resolves over too: a qualifier or a fragment's name
  reaches the binding in scope where it is spelled. File-wide lookup failed silently both ways:
  gh-ocannl-1063's inventory recorded another leg's `let body_begin` text for a pin (a labelled
  parameter of that name read as the literal), and staging#855 lost the pins of two same-named
  bindings outright, equal values included. Predicate callees resolve to the lexical function
  binding too (gh-ocannl-1150), since their body text must not come from a same-named helper in
  another scope. Source provenance also resolves identifiers at each use (gh-ocannl-1186);
  emitter value aliases and buffer destination seeds retain their existing conservative name model.
  `codegen_text_scan_cases` controls each spelling against the old resolution.
- Predicate parameters use the emitter destination lookup too (gh-ocannl-1150): labelled text
  and source parameters match by label, while only unlabelled parameters consume positions.
  Keep every text test in a predicate, since `residency_holds` pins several caller-supplied markers;
  choosing one loses the rest even after the parameter reader learns labels. Optional defaults
  must not stop that reader before a later source parameter.
  `Codegen_text_scan.scoped_provenance` carries generated-source evidence, symbolic parameter
  dependencies and unresolved buffer evidence through one lexical rule for bindings and uses
  (gh-ocannl-1186). Rebinding a parameter through normalization preserves its dependency; replacing
  it with unrelated text keeps uncertainty without attributing the caller's fragment. A helper
  sourcing its own generated text does not become an unresolved forwarding wrapper, even when
  unused. Validated calls seed the specific helper parameter, so same-named parameters elsewhere
  cannot borrow that evidence. Captured generated bindings use the same scoped rule. A helper
  returning generated text from its own read supplies that evidence independently of its routine
  argument; a pure helper substitutes only the formal dependencies of its returned value at
  each call. Buffer-returning helpers keep inputs used by earlier writes at an explicit uncertainty
  boundary; the scan does not model memory effects. Known ordinary replacement and unresolved
  callback input have distinct uncertainty states. A named fragment requires generated-source
  evidence; an untraced callback parameter alone keeps only the partial mark. Unary callbacks
  for `List.iter`, `map`, `exists`, `for_all`, `filter`, `filter_map` and `count` receive generated evidence
  only from their actual collection argument, with unresolved execution. A leg claim's make
  function (`in_leg`/`in_legf` qualified by `Verdict` or `Claims`, or unqualified and bound by
  nothing in the file) receives its claim's label, a known plain string, so its body reads as
  ordinary code (gh-ocannl-1703). Other combinators and callback shapes remain unsupported. Symbolic components
  of opaque aggregate inputs and results flowing through explicit writes are uncorrelated, so
  substituting a generated argument cannot pretend that the selected component contains it.
  Discarded ref, field, array and byte writes preserve source dependencies as uncertain effects;
  this is a boundary, not a memory model. Independent returned evidence survives that boundary.
  Completed predicates return ordinary booleans, rather than carrying their callable identity into verdict bindings.
  A composite fragment retains its literal context and every caller-supplied component in the
  inventory. An omitted optional marker uses its default only when a later positional argument
  selects it; static optional forwarding resolves `None` to the selected default and unwraps
  `Some`. Dynamic forwarding keeps the expression and possible selected default, with uncertainty
  when they can carry source. `function` case inputs use a synthetic positional formal and the
  same match-payload rule. Explicit tuple/record components bind their own source evidence in
  lets and matches; opaque aggregate destructuring keeps possible fragments with a partial mark.
  A match over a `try` value retains successful result evidence, while handler and exception
  patterns remain ordinary. A partial call keeps unsupplied symbolic dependencies explicitly
  unresolved; the scan does not rebase closure argument positions. Rebuild call aggregates at each
  fixed-point step, so an earlier unresolved input cannot poison a later validated chain.
  Predicate body literals pass the same call-site source check as caller markers.
  Ordinary forwarding wrappers, mutation, partial applications and unfollowed callbacks remain
  explicitly partial; this includes ordinary-source calls, which contribute no fragment. Known
  local calls can validate captured parameters and callbacks; unknown higher-order calls keep any
  independently validated fragments alongside their partial mark. A parameter the callee writes
  into a cell is recorded (`provenance.stored`), so substituting it does NOT discharge `Mutation`
  (gh-ocannl-1237): an argument that only the cell returns joins uncorrelated, one also returned
  directly keeps its pins beside the mark, and ordinary text adds nothing. A wrapper forwarding
  its parameter to such a callee stores it too, and a call made only for its effect (a discarded
  `store z;`) is a write of what it stores. The cell operators are
  `!`, `:=`, `Array`/`Bytes` `get`/`set` and field writes, bare or under a `Stdlib` qualifier.
- **What no file-local rule can follow now says so.** A buffer is where generated text lands with no
  name to carry it, and the ways to fill one do not end (a wrapper reaching its parameter through a
  local binding, PPrint's own `ToBuffer` renderers, a buffer in a record). So a substring test whose
  haystack reads a `Buffer.contents` this scan never saw filled — directly, or through the bindings
  the read travels along, by the same scoped provenance rule — marks that file's itemisation
  **partial** rather than dropping the fragment silently — the file is listed, the fragment is
  unnamed, and the inventory says which.
- A test that classifies COMPILER-PLAN text rather than generated text marks that one binding with
  `[@@ocannl.codegen_text.compiler_plan]` (gh-ocannl-865). The inventory excludes text tests only
  inside that binding: the annotation is not file-wide, so a real generated-source pin elsewhere in
  the same test remains visible. `codegen_text_scan_cases` controls both directions — an unannotated
  compiler-plan classifier is still reported as partial, and an annotated classifier beside a real
  generated-text assertion still reports the real fragment.
- **An `open` (or `include`) that hides a route is refused, not approximated — over its own scope,
  and never over a name bound where it is spelled.** A structure open governs the items after it,
  `let open M in` its body, and a nested structure's opens die with it. A name the file binds is
  struck from a refusal wherever that binding is in scope — only there, since gh-ocannl-1079:
  `open Ir.Low_level` followed by a local `let to_doc` is valid code calling the
  local function, and refusing it would red the build for everyone, where a refusal not made is one
  more member of the residue the partial marker covers. Judging the file's opens against the file's unqualified uses cross-products the two, and a
  false refusal on valid code is a red build for everyone (Codex round 3 on staging#487). Every route is attributed by the
  qualifier at the call site, so `open Test_utils.Generated` followed by a bare `read` reads exactly
  like a local function of that name and drops the file out of the census. `Codegen_text_scan.
  rejections` fails the inventory on that spelling — for the artifact readers and for an emitter's
  own module alike — rather than tracking opened scopes, which is one more approximation with one
  more edge. Write the qualified spelling, which is what every test already does.
- Each golden in that inventory carries the family that must re-record it, and the DECLARING
  extension wins over the markers: a `.hip.expected` spells CUDA's `__global__` launch vocabulary
  and is still HIP. A fragment the scan cannot name at its call site — text a helper computes —
  marks that file's itemisation partial rather than dropping it, so the file is still listed and
  the re-run is still called for. The mark names the provenance boundary that stopped it
  (gh-ocannl-1210), spelled in the golden as: computed fragment, untraced callback, forwarded
  parameter, opaque aggregate component, unsupplied application, mutation or unseen buffer write,
  rebound parameter, unvalidated haystack. The last is usually benign — the scan's safe default for
  a source-taking helper called on ordinary text such as a backend name. The category is carried by
  the shared provenance result (`Codegen_text_scan.boundary`), never decided at the print site: a
  new uncertainty source goes through `uncertain`, which always names one (`boundaries_of` raises
  on a level that names none), and a new constructor earns a single-boundary case in
  `codegen_text_scan_cases`, whose coverage claim quantifies over the derived `all_of_boundary`.
  Markers are read only outside `Verdict` claim lines, since a claim label is prose ABOUT a kernel and freely quotes its vocabulary ("padded GPU intrinsics fire
  against the threadgroup fragment").
- A golden tagged `[derived] beside <source>` got in by neither route. The markers describe whole
  dumps, and a golden can hold emitted text in fragments instead — a table whose columns are the
  `%cd` and C-style spellings of one constant, a census of the schedule decisions a kernel was
  built from. Such a file is a member when the test beside it demonstrably reaches generated text
  AND its golden holds something other than that test's own verdicts. That last condition is what
  keeps the rule useful rather than noisy: a boolean column does not move when codegen does, so a
  schedule test's all-`true` golden stays out while its source stays in.
- `codegen_text_inventory` is one of a family of **checklist inventories**: a scan whose golden
  lists every file a given kind of change must revisit, derived rather than hand-kept.
  `provenance_tag_inventory` (gh-ocannl-1015, staging#836; every family since gh-ocannl-1081) is
  the other, listing every file that cites a placement-provenance tag, for a change to a tag or to
  pipeline order. The trap they share: a text inventory reads the whole tree, its own cases file
  and golden included, so a fixture spelling the very text it matches becomes a member (or a
  refusal) of the real run. Spell fixtures in pieces (`provenance_tag_scan_cases` still spells
  its constructor so), and exclude the scan's own golden, as `codegen_text_inventory`'s `excluded` list does — else a
  promote takes two rounds to converge. Where EVERY token of a shape is read, as the provenance
  inventory reads every tag-shaped token so that a retired one is refused, pieces cannot hide an
  invented fixture tag: that scan excludes its cases file and golden by name instead. That shape
  (digits, a colon, a kebab-case reason of two or more words) is reserved for tags in every file
  read, shell included: a `case` label of that shape reads as a stale citation, so spell such a
  pair status last (`some-state/0`). The scan exempts no syntax, because a label branching on a real
  tag is a citation a retirement has to find.

## Aliases and goldens

- Backend-disabled targets from `affected` or the codegen inventory are handled by
  `test-run.sh run` and `start` (gh-ocannl-2161): every skipped operand is printed,
  counted in the digest and stored in `disabled-aliases` with `disabled-alias-count`.
  An all-disabled alias batch refuses with exit 2. The reader recognizes explicit
  rule/alias and test/tests declarations, inline-test libraries, logical
  subdirectories, and `enabled_if`
  boolean combinations of literal equality/inequality and `OCANNL_BACKEND` environment
  interpolation. Unsupported expressions stay with Dune; includes or an unreadable
  census disable this filtering. Cram or interpolated alias names keep their
  directory's aliases with Dune. Every matching owner must be disabled, so an active
  descendant keeps a recursive alias. Replay modes (`repeat`, `fresh`) refuse
  recognized disabled aliases rather than replaying a partial set. Leg 120 of
  `tools/test-test-run-options.sh` uses real Dune: generated disabled aliases disappear
  from its listing, while explicit disabled rule aliases remain listed and empty.


- Every rule that diffs a golden — the repo-wide scans, the codegen snapshots, the config-precedence
  rules, the ppx-output diffs — carries its own `runtest-<name>` alias (gh-ocannl-726), so
  `dune build @test/operations/runtest-verdict_ratchet` runs that one test, applies its diff and is
  promotable, and a misspelling exits 1 (`Alias … is empty`) rather than doing nothing. The
  repo-wide scans additionally share `dune build @test/operations/scans` (gh-ocannl-703): the family
  runs in seconds when cached (about half a minute after a source edit), and WHEN to run it is stated once, in `AGENTS.md`'s Testing bullet on the scans,
  with the test-only edits it covers explained under *Repository scans* above. Two properties of dune shape how those aliases are written, and both are silent
  when got wrong:
  - **A rule attached to two aliases makes building EITHER build BOTH.** So a golden-diff rule sits
    on `runtest-<name>` *alone*, never on `(aliases runtest runtest-<name>)` — that spelling type-checks,
    passes its own test, and quietly puts the whole directory behind every per-test alias (measured:
    `@test/operations/runtest-verdict_ratchet` ran the entire `test/operations` suite). What keeps
    those rules in plain `dune runtest` is an `(alias (name runtest) (deps (alias runtest-<name>) …))`
    stanza per directory — the same shape staging#431 gave `slow`. `env_var_deps` fails on a member the
    stanza omits, and on a golden diff attached to `runtest` itself.
  - **`runtest-<name>` is dune's own namespace for `(test)`/`(tests)` stanzas and inline-test
    libraries** (dune >= 3.20). Reusing such a name for a rule is a dependency cycle as soon as the
    rule also names `runtest`, and is confusing in any case, so a hand-written per-test alias names
    the GOLDEN rather than the stanza: `runtest-zero_out_local_decl-unoptimized` beside the
    `zero_out_local_decl` test, `runtest-test_ppx_op-ppx` beside `test_ppx_op`. Adding an action to a
    generated alias is legal, and the per-directory `runtest-env_spelling_gate` rule does exactly
    that so every per-test rule can depend on the ambient gate -- and is the one rule `env_var_deps`
    lets share such a name, recognized by its `(universe)` dependency rather than by name. `env_var_deps` compares the two
    literally: the alias must begin with the golden's name, since what a reader has in hand when
    they reach for the alias is the golden that just failed, and an alias that renames it is one
    they construct empty.
- **The ppx goldens pin four fixtures; a ppx change's reach over the tree is
  `tools/ppx-output-diff.sh BASE HEAD`** (gh-ocannl-1886). `test/ppx/*_expected.ml` hold the
  expansion of the sources beside them, so a change that alters any other module's expansion passes
  them. The script builds every `.pp.ml`/`.pp.mli` dune defines at both revisions, each in a
  throwaway worktree, prints them back to source and diffs them; stderr tags each file `EXPANSION`
  (its input unchanged: the preprocessors' own reach), `SOURCE`, `ADDED` or `REMOVED`, and
  `-- DIR...` narrows it. The whole tree takes about 80 s on mac-studio (warm dune cache), as a
  `--cpu` slot batch. It sees tracked modules only: a scan table's snippet (the `let%op` arm
  gh-ocannl-1787's fix silently changed) is expanded by `tools/typecheck-scan-cases.sh` alone, and
  over that fix's merge (staging#1599) the script reports its two edited files as `SOURCE` and no
  `EXPANSION` anywhere. Harness: `tools/test-ppx-output-diff.sh`.
- **Inline `%expect` tests get a per-MODULE alias too** (gh-ocannl-1037): the alias dune generates
  for an inline-test library, `runtest-operations_tutorials`, runs every module, so one tutorial
  checked on a GPU box used to cost the directory. Each module holding inline tests has a rule on
  `runtest-<module>` running the library's generated runner,
  `%{dep:.<library>.inline-tests/inline-test-runner.exe} inline-test-runner <library> -only-test
  <module>.ml -source-tree-root %{workspace_root} -diff-cmd -`, then `(diff? <module>.ml
  <module>.ml.corrected)` — the flags and the diff are the ones dune's own action uses, so the
  result promotes the same way. Four mechanics. The runner is an ordinary target at the path dune
  generates it under (`dune build test/operations/.operations_tutorials.inline-tests/inline-test-runner.exe`
  builds it), so a `%{dep:…}` reaches it from the library's directory. `-only-test` (not
  `-partition`) is the restriction because the runner fails when an `-only-test` matches nothing
  (`the following -only-test flags matched nothing`), so a renamed module cannot leave its alias
  passing on an empty run; `-partition` with an unknown name runs nothing and exits 0. The rule
  takes `(sandbox always)`: dune's own inline-test action is sandboxed (lang >= 3.5), and both
  write `<module>.ml.corrected`, which unsandboxed would race when one build requests both. And
  these aliases are deliberately NOT aggregated onto `runtest` — the library's generated action
  already runs every module there, so listing one runs its tests twice. `env_var_deps` derives the
  obligation from the library's module set and its sources (`let%expect_test`, `let%test`,
  `module%test`): it fails on a tested module without its rule, on a runner rule whose
  `-only-test`, alias and golden do not name the same one module, and on such an alias in the
  `runtest` aggregate, while the golden-diff naming rule above covers it like any other.
- **A per-test alias runs its own golden and no other.** `runtest-NAME` does not reach a variant
  `runtest-NAME-VARIANT` holding a golden of its own (`NAME-VARIANT.expected`, or
  `NAME-VARIANT.ll.expected`): the same executable again, under another setting. A hand-named batch
  that changes what the test prints and promotes leaves those goldens stale, and only CI says so.
  `plan`, the launch and the digest name such siblings when the batch does not (`sibling-goldens`
  in the run): add them and run again before promoting. It is a note, never a refusal, and adds
  nothing to the batch. `aliases PATH` lists a test with its variants, and `affected`'s list already
  holds them, since a variant has its test's stanza as an owner; a base golden changed alone reaches
  only its own alias, because no sibling's rule reads it.
- **A `(test)` stanza can only diff the one `<name>.expected` beside it**, so a test whose output
  legitimately differs per backend converts to an `(executable)` plus a diff rule that reads the
  resolved backend name and diffs `<name>-%{read:../config/ocannl_backend.txt}.expected`. That is
  the gh-ocannl-700 shape (`micrograd_demo_logging-<backend>-0-0.log.expected`). Three mechanics come
  with the conversion. The resolved name is NOT produced per directory: `test/config/dune` — the
  directory every test directory already copies `ocannl_config` and `env_spelling_gate.ml` from —
  holds the `ocannl_read_config` executable and one rule per resolved file, `ocannl_backend.txt`
  and `ocannl_backend_extension.txt` (the latter naming the emitted-code suffix an artifact carries,
  which is what `test/operations`' `top_down_prec`/`zero_out_local_decl`/`test_where_precision`
  rules read). A directory adopting a per-backend golden writes `%{read:../config/…}` and copies
  nothing; the reader normalizes the deprecated aliases so `multicore_cc` finds the `multidev_cc`
  golden, and one resolution serves the whole suite because every directory copies the same
  `ocannl_config` and shares the environment and command line. `%{read:…}` works in a rule's `deps`
  as well as its action, which matters
  because the action must be wrapped in `(no-infer …)` — otherwise `with-stdout-to <name>.actual`
  registers a target and plain `dune build` (@all) runs the training that the `(test)` stanza only
  ran under `runtest` — and `no-infer` also drops the dependency `diff` would have inferred on its
  golden, so the golden goes in `deps` by hand. And the new `runtest-<name>` alias is a build entry
  point, so the directory needs a `runtest-env_spelling_gate` rule for its output rule to depend on
  and an
  `(alias (name runtest) (deps (alias runtest-<name>)))` stanza, both of which `env_var_deps`
  checks.
- Splitting a golden per backend is a decision about WHAT the golden holds, not a formatting choice,
  and the bar is high in both directions. Its cost is a golden only some machine re-records — cc
  and metal on the reference Mac, multidev_cc, cuda and hip only on the sweep's minix and rog-nv
  lanes (a local multidev_cc run refreshes nothing the sweep then checks on the Mac) — so a
  codegen change that moves the output leaves the members no local run touches stale until the daily
  sweep says so. That is the gh-ocannl-700 lesson, and it makes a split worth paying for only where
  the difference is a genuine backend fact. **A member that keeps diverging from its siblings AFTER
  the split, or diverging from itself between runs, is evidence the split was a misdiagnosis**:
  `transformer_names` was split per scheduler on the reading that `Multidev`'s execution order may
  move a training trajectory, and the real cause was `multidev_cc` launching kernels on whichever
  static index the host had raced ahead to (fixed by the dispatch-time snapshot in
  `Backends.Add_device`; `test/operations/async_launch_bindings` pins it). One golden serves all five
  backends again. Before splitting, get a run-to-run repeat of the diverging member: a stable
  divergence can be a backend fact, a moving one never is.
- Pinning output no claim can replace is the other half of that decision. The gh-ocannl-725 rule
  relocates a device-produced float to stderr only when a stdout claim can be written that FAILS on
  a wrong value, and no cheap claim over sampled text does — "non-empty, from the training alphabet,
  properly terminated" passes on an untrained model sampling gibberish, while the name-likeness of
  `holern`/`cern` is the legible canary that the model learned name structure. So `transformer_names`
  keeps its three names pinned, and the burden falls where it belongs: on the backends being
  deterministic and agreeing.
- Every per-backend golden family covers the backend vocabulary derived from
  `Backends.all_of_backend`; `backend_golden_family_scan` derives active family templates from the
  dune rules that read `ocannl_backend.txt`, relates them to the expected-file census, and fails
  with the derived and present lists before dune can report a raw missing-rule error on the absent
  backend (gh-ocannl-802). A member copied from a shared golden rather than recorded on its own
  backend carries a marker inside the dune rule that references its family:
  `; ocannl-golden-recorded-on: <member>.expected <- <backend> -- <reason>`. The scan validates the
  declaration and family relationship on top of `Dune_stanza_scan.contained_marker_contract`, so
  its outer grammar, containment, wrong-stanza refusal and sentinel accounting are the same contract
  as `ocannl-backend` rather than a second implementation. The scan prints the member into its
  golden without failing on it; remove the marker after a run on the member's own backend re-records
  it.
- Gating a new slow test is the same conversion with `slow-` names: replace the `(test)` stanza
  with an `(executable)` plus a `(rule (alias slow-<name>) …)` that runs the exe and diffs
  `<name>.expected`, put `(alias slow-env_spelling_gate)` first in the rule's `(deps …)` so the
  ambient gate runs before the test, and list `(alias slow-<name>)` in the directory's
  `(alias (name slow) …)` aggregate — `test/training/dune` is the pattern, and `env_var_deps` fails
  on a rule the aggregate omits, since `dune build @slow` would otherwise skip it silently. The
  action needs `(no-infer …)` here for the same reason as above: an `.actual` registered as a build
  target puts the slow run on plain `dune build`'s `@all`. Regular and `@slow` training actions take
  the global `/ocannl_training_test` lock so their OpenMP pools never overlap; `test/operations/cpu_parallel`
  stays unlocked on purpose. `dune build @runtest @slow` runs both families, and `@check` compiles
  the slow executables, so they cannot bit-rot while off the default path.
- Two focused aggregates sit beside `scans`, built the same way and answering a narrower question
  (gh-ocannl-783): `dune build @metal-codegen` runs the Metal-pinned tests — the executed Metal-only
  guards and the emitted-MSL structural ones — and `dune build @lifecycle` runs the
  resource-lifecycle probes, the tests that drive `Ir.Resource_fault_injection` or read
  `Ir.Alloc_census`. Each family alias is spelled identically in `test/operations/dune` and
  `arrayjit/test/dune`, which is what makes the root-level `@<family>` run both halves: `dune build
  @foo` builds the alias in the current directory and every directory beneath it. Both run in
  seconds against warm executables, which is the point — the backend-wide directory run that was
  previously the only way to reach these legs is neither.
  - Metal membership comes from the stanza's `; ocannl-backend:` marker naming `metal`.
    Lifecycle membership uses an explicit structural contract (gh-ocannl-798): put
    `; ocannl-lifecycle: <name>[,<name>...] -- <reason>` inside the `(test)`, `(tests)`,
    `(executable)`, `(executables)` or inline-test `(library)` stanza. Each name must be one that
    stanza builds, with no empty or repeated entries, and the reason must have at least two words.
    A plural stanza can declare one unit without claiming its neighbours. The library dependencies
    must reach the actual instrumentation owner: `Lifecycle_scan` derives its names from the
    library owning both instrumentation modules in `arrayjit/lib/dune`, then follows local Dune
    library dependencies. A missing owner, inert library path, ambiguous name, or unsupported
    dependency form with no established owner path is a clear error. Marked stanzas require literal
    library lists; optional forms beside a concrete transitive owner path do not erase that proof.
    Marker strings, comments outside a stanza, rule markers and duplicate declarations are errors.
    The executable's runner supplies its aliases, so a declaration on an executable with no reached
    runner fails the same family-completeness check. Membership is declared intent and link
    capability; it does not prove that a probe exercises instrumentation. Review owns that assertion.
    The broad `arrayjit.ir` library also serves ordinary tests, so linking it alone declares no
    lifecycle intent. OCaml opens, aliases, functors and shadowing no longer decide this family.
    `Config_key_scan.module_references_in_source` remains a separate consumer for the schedule
    pipeline's configuration-key derivation (gh-ocannl-1149), outside this membership contract.
    The derived set remains a floor: a family may also include probes such as
    `arrayjit/test`'s `test_slab_free_on_grow`, which drives a mock raw backend.
  - Writing a new family means giving it the same treatment. The derivation has to be a property the
    member stanza declares for an INDEPENDENT reason (a backend marker, a module it uses, a glob it
    writes). Where no distinct library separates the capability, an explicit declaration may
    replace source resolution, with structural ownership and link-capability validation as above. A family
    alias is a build entry point, so it depends on `(alias runtest-env_spelling_gate)` first, which
    is why `arrayjit/test/dune` carries that gate rule and an `(alias (name runtest) …)` stanza
    aggregating it back.
  - `env_var_deps --control` puts both derivations to trees this repository does not contain: a
    member with the family stanza omitted reports and exits 1, the same tree with the member listed
    passes, and a stanza neither derivation claims is asked for nothing. Extending the check without
    extending that control leaves the new arm able to stop deciding anything.
- A record with `[@@deriving sexp]` makes every `.expected` file that prints the parent a hidden
  consumer of its FIELD NAMES, and `rg "\.field_name"` over sources is vacuous against that (sexp
  prints `(field_name value)`, not member access). Before claiming a rename has no serialization
  consumers, grep the sexp shape: `rg -F "(field_name " --glob '*.expected'` (the trailing space
  disambiguates longer identifiers). Budget the resulting promote as expected work, in its own
  commit, after diff-confirming the delta is rename-only.
- Before broad testing of a config-gated path, grep whether any test or test config enables the
  gate: a path no test turns on is exercised by none of them, however wide the batch, and the
  evidence is a test that sets the key (or `OCANNL_<KEY>=...` on the run), not a green suite with
  the gate at its default. The full regular and `@slow` suites are reserved for cross-cutting
  changes, where `affected`'s list (below) spans over half the suite anyway; everything narrower
  runs the aliases `affected` ranks and the `slow-` members it names.
- **Which per-test aliases a change reaches is dune's answer, not a guess**:
  `tools/test-run.sh affected --since BASE | REV | REV..REV [-- PATHSPEC]` (or `--files PATH...`)
  prints them, one `@DIR/runtest-NAME` (or `train-`, `slow-`) per line, ready for `tools/test-run.sh
  run build` or `fresh`; stderr says what each changed path reached and gives per-directory counts.
  The pre-push batch is `prepush --since <base>` (below), which runs it, where `<base>` is the
  remote-tracking `master` of `lukstafi/ocannl-staging` (`git remote -v` names it: remote names
  are not the contract): it compares the worktree with `git merge-base HEAD <base>` (gh-ocannl-1379). A bare REV
  that is not an ancestor of HEAD is warned about, since the base branch's tip would also count
  every commit it gained since (gh-ocannl-1235: a branch eight commits behind listed 446 aliases
  for a 69-alias change). It reads dune's own description, never a hand-kept map: `dune describe
  workspace` (modules and library edges), `dune describe external-lib-deps` (the ppx-rewriter
  edges and the `tests` stanzas the first omits), `dune show aliases`/`targets`, and `dune
  rules` for each alias's own targets and each generated module file. Those dependencies are how
  a scan that READS a changed file (an `.ml`, an agent note, a script) is found without linking
  it, and how a generated module (`.mll`, `.mly`, a `(select)` choice, a generator's header) or a
  C stub is traced to its stanza. dune does not dump an alias rule itself (Dune
  mechanics, the introspection bullet), so `affected` hands it each one in a
  form it does print (gh-ocannl-1541): a scratch copy of the sources, the PROBE, whose dune files are
  dune's own canonical text (`dune format-dune-file`) with every alias rule that declares no targets
  rewritten into a TWIN -- the same rule writing a file of its own, its aliases kept defined by an
  `(alias (name X) (deps TWIN))` -- and every test given a golden, so dune makes its output rule.
  `dune rules` on a twin is what the action reads, dune's own evaluation of `%{dep:}`, `diff`
  inference, nested or templated `chdir`, `%{workspace_root}` paths, globs and needed aliases. The
  converter is a strict allowlist (heads, alias/targets/action fields, literal names and paths);
  an alias it does not vouch for stays BLIND and is listed as before -- for any changed path in its
  directory but an owned module source, and for a changed non-build input anywhere, with every
  alias needing it. What stays blind here: the inline-test library aliases (`runtest-LIB`; dune 3.20 has no
  `dune show tests`, so tests are read from their stanzas' literal names), and anything with an
  `(:include)` or `dynamic-run`, whose
  dependencies `dune rules` does not print (Dune mechanics, same bullet). A `(package PKG)`
  dependency is dune's `.PKG-files` alias, read through `dune rules @.PKG-files`. Before the probe,
  `runtest-kernel_attribution` (a run-and-diff rule, gh-ocannl-1431) and every `(test)` without a
  golden were blind, and every golden pair needing the blind `runtest-env_spelling_gate` came with
  them: 184 aliases for a docs edit. A `dune` file edit is compared across revisions in an OLD
  probe (that file at the old revision): the rules of its directory's targets that differ, twins
  included, plus the library/executables/tests stanzas whose canonical form changed (flags live in
  object rules not asked), by literal names dune lists; a changed form of another kind (`env`,
  `subdir`, `include`, ...), a renamed module the old file cannot describe, or `--files` (no
  revision) keeps the whole directory, and says why. A hand parser of `(deps)` and stanzas
  (staging#1336) did not converge in review: five rounds of missed forms, each an under-reach; the
  twins leave every dependency form to dune. The probe costs about 10 s a run (cold build
  directory, compiles served by the dune cache) and a compared dune edit about 30 s;
  `OCANNL_TOOL_AFFECTED_PROBE=0` skips it, reading every alias action blind -- and missing what
  only a twin shows (an executable of another name a run-and-diff rule runs: 19 aliases on a
  `low_level.ml` edit), so the default keeps it. The header states the boundary; what falls outside is
  reported on stderr, and a changed
  `.ml`/`.mli` no stanza of this build owns (a deleted module, a CUDA-only one on a box without
  cudajit) exits 3. Two portability points (gh-ocannl-1389): only the file
  `dune rules -o` writes is the answer, since on a cold `_build` printing a rule builds the
  `%{read:…}` file it depends on (`ocannl_backend_extension.txt`) and that action's chatter lands
  on dune's own streams (a refusal names the atom and keeps the whole answer in
  `$TMPDIR/affected-dune-rules.PID.txt`; the one sighting reported after that fix ran a branch cut
  before it, gh-ocannl-1431); and a library edge between two stanzas this build does not compile
  is skipped, since `external-lib-deps` gives public names while such a stanza may use a local one
  (on Linux, `metal_backend` requiring `metal_math_api_runtime`).
  `tools/test-affected-tests.sh` pins it on a fixture project, a negative control per reach path.
  - Reach is link-level: a module reaches its whole library, and every stanza requiring it.
    So a change in `arrayjit/lib`, `tensor/` or a widely linked `test/support` library reaches
    much of the per-test alias set, including `arrayjit/test` and the `test/training` members:
    a cross-cutting change in AGENTS.md's sense, and the list is the reach CI will exercise. The
    three misses that filed the issue are all inside it: `Cost_model` (staging#1011) lists
    `@test/operations/runtest-cost_model_template`, `Register_tile` (staging#1035)
    `@arrayjit/test/runtest-test_vectorized_codegen`, `Schedule` (staging#1040)
    `@test/training/runtest-fused_classifier`. A narrower "tests that name the module" cut is no
    substitute: `test_vectorized_codegen.ml` never names `Register_tile`.
  - So that cut ranks, never drops (gh-ocannl-1431): an alias reached only by linking, none of
    whose owners' own code (its sources, a project ppx's, and of the private libraries it links
    the top-level definitions that code names, comments blanked) names the changed module or a
    re-export of it read through the module making the re-export -- nor, when git's hunks place
    every changed line in top-level definitions, one of those definitions -- is LINK-ONLY: printed
    last, named on stderr, left off by `--referenced-only` (a flag of the whole call: any position
    before `--`). A changed line inside a `type` touches the type's name, the constructors and
    fields that line names, and what ppx derivers make of the name; an attribute line, the whole
    module. What made the cut blunt before gh-ocannl-1651: a helper library counted whole, so a
    test calling one `Ll_test` function named every module any of its ~90 definitions or comments
    named; and a re-export counted bare, so `module Lazy = Utils.Lazy` in `hip_backend.ml` made
    every `Lazy.force` a mention of Utils. Three readings narrowed it further (gh-ocannl-1765).
    A change to `optimize_proc`, which every compile runs and no test names, still reaches tests
    by linking -- the reason the default keeps link-only aliases. `utils.ml` as a whole reaches
    every config-reading alias through `ocannl_ambient_env`. A dune-only library split ranks by
    the moved modules' names. Read current counts from `tools/test-run.sh affected` rather than
    preserving a census here.
    A re-export is read through its re-exporter: `Tn` names `Tnode` where `Context.Tn` qualifies it
    (a local `module C = Context` too), or in a file that opens or includes the re-exporter (`open`,
    `let open`, `X.( )`), or in the re-exporter's own source -- an open in a helper opens nothing in
    the test calling it, and any other token of the re-exporter no longer counts (`Tn` beside a
    `Tensor.print`). An `include M` in X re-exports into X itself, so naming X is the reference
    (`Autotune.sketch_seed_params` names `Sketch_families`). A `(select)` module counts by every
    arm's file (a `tests` stanza, read from its dune form), and a described stanza's copied module
    (a select arm, a `copy#`: a rule that runs nothing and reads one OCaml file) by the file it
    copies; both were "code not all known", so always first (12 of the 21 code-referenced owners
    of the `tnode.ml` case below). And a changed line holding no code -- blank or comment only,
    outside every literal -- touches no definition: the blank line or doc comment above an added
    definition had touched the definition above it (the `walk_scalar` sighting), and a
    comment-only edit now ranks every linking alias last -- unless a delimiter changes how a line
    git shows unchanged reads (a definition commented out between added `(*` and `*)` lines, or
    brought back), which changes the module as a whole. One added function in `tnode.ml` went
    from 628 aliases to 24 under `--referenced-only`, all of them scans reading the file. The re-export reading alone moved no count on the real edits tried
    (`tnode.ml`, `indexing.ml`, `ndarray.ml`, `assignments.ml`, `low_level.ml`): every owner it
    stopped counting also names the module itself, or is reached through `ppx_ocannl`'s sources,
    which name `Assignments`, `Indexing` and more for every `%cd`/`%op` user (the boundary above).
    When the first tier still spans over half the suite, stderr says CROSS-CUTTING: the PR's CI
    run owns the full list then, but for its `slow-` aliases, which no CI job runs (CI builds
    `@train` or `@default @runtest @bin-smoke`; `@check` only compiles `@slow`), so the local run
    is those plus the tests exercising the change (AGENTS.md).
    A list spanning over half the suite is CROSS-CUTTING too, whatever its first tier
    (gh-ocannl-1996): link-level reach lists 815 of the 825 per-test aliases for any `arrayjit/lib`
    module (`low_level.ml`: 379 first, 436 link-only; `metal_backend.ml`: 27 first) and 631 to 636
    for one of `tensor/` or `lib/` (`autotune.ml`: 307 first), and read `ok` they made `prepush`
    a near-full suite no shared box runs, so six workers of one wave hand-picked their sets
    against it. A separate threshold on the printed total, not link-only aliases counted into the
    first tier: that tier still says what the ranking could narrow, and the line for this case
    names it as where the local run starts (`affected --referenced-only`, which prints that tier
    alone and is judged on it). Both thresholds are over half today; the lists found were under
    40% (`test/support/ll_test.ml`, 329) or over 76%. So `prepush` now runs for tests, test
    support, tools and docs, and refuses for a library module.
    The machine-readable form is the trailer (gh-ocannl-1921): whenever the list is printed, the
    last line of stderr is `affected: verdict=V`, V one of `ok`, `cross-cutting`, `incomplete`
    (a changed path is UNMAPPED, exit 3) or `cross-cutting,incomplete`. A caller reads that line,
    never the prose; both conditions are decided in one place, `verdict` in `affected-tests.sh`,
    with one function per threshold (`first_tier_wide`, `list_wide`), which is where a different
    cross-cutting rule goes; the `verdict trailer` leg of `tools/test-affected-tests.sh` calls it
    on the counts above.
    The contract: LINK-ONLY is a heuristic ranking,
    and `--referenced-only` a deliberate trade (faster, with a possible miss) for iterating
    mid-change, except the explicit CROSS-CUTTING first-tier local run below. The pre-push run
    is `prepush`: the default list without its `slow-`/`train-` members, unless the verdict is `cross-cutting`, when `prepush` refuses
    and the PR's CI run owns the list.
  - `--why ALIAS` (repeatable; `--why all` for every listed alias) explains a rank on stderr, in
    `affected: why @ALIAS:` lines, stdout untouched (gh-ocannl-1711): a verdict (REFERENCED,
    BLIND, LINK-ONLY, NOT LISTED), then each reason with the changed path behind it -- a dune file
    over its directory, an own rule reading the path, a needed alias, and per reached owner the
    library chain to the stanza the path reached plus the token that made it referenced, traced
    back through helper definitions to the owner's own source (`` `H` at t8/hlp.ml (definition
    `names_core`) <- `names_core` at t8/t_hnamed.ml (its own source) ``), a re-export with the
    module making it, or why its code is not all known. It reads the tiers' own facts (one
    `owner_names` serves both the ranking and the explanation), so it cannot disagree with the
    list. Reach for it before instrumenting the script: gh-ocannl-1651's causes took four
    instrumented copies to find. On a three-definition `tnode.ml` edit (720 aliases, 37
    referenced) it showed the shape of the residue: of the 21 owners referenced by code, 12 held
    code not all known (a generated test module, a `(select)`: read through their arms since
    gh-ocannl-1765), 6 named a changed definition with the module (4 listed with `Tn` beside a
    re-exporter such as `Context` or `Tensor`, but each also names `Tnode`, so the stricter
    re-export reading keeps them), 4 of them through a `test/support` helper definition.
  - `affected --gpu-gated` (gh-ocannl-2074) filters that same ranked reach by declarations
    in an alias's owner sources (`binds_hardware_grid`, `on_gpu`, or `leg`/`in_leg`/`skipped`
    beside `on_cuda`, `on_hip`, `on_metal` or a literal GPU backend) or a GPU `ocannl-backend` marker inside its
    owner or own rule.
    Use `--referenced-only` with it for direct mentions; omit that option to retain GPU-gated
    link-only consumers too. Stdout is aliases ready for a backend run; stderr names the gate
    evidence for exactly those aliases. This is the GPU-specific minimum: ungated every-backend
    tests in the unfiltered list also exercise GPUs and are excluded here. The unfiltered verdict
    remains, so a narrow GPU list does not waive CROSS-CUTTING.
    This is by-reading, at file granularity: comments are ignored, but unused names or a CPU leg
    beside a GPU literal can overselect; computed gates, generated sources and gates hidden in
    helpers can be missed. It neither proves runtime coverage nor infers which GPU can run a leg.
    `tools/test-affected-tests.sh`'s GPU-gated reach leg pins a changed function's callers,
    CPU and unrelated controls, stanza-local markers and a disabled-discovery negative control.
  - The shell harnesses are no dune aliases, so `affected` reads them from their own text
    (gh-ocannl-1900; the boundary is `tools/affected-tests.sh`'s header, The harness legs): each
    member `tools/test-harnesses.sh --list` names reaches what its program (it and the scripts it
    sources or imports, to a fixpoint) names, with the program of each script named; what a named
    script names in turn is not followed, so a harness driving a script through another names it
    itself. stderr gives each changed path's harnesses with the step that reached each, then
    `affected: harness: CMD` lines, counted in the closing summary when there are any (`..., and N
    harness command(s), named above`, gh-ocannl-1929); `--harnesses` prints only those commands,
    on stdout, and asks dune nothing (under a second here). A header declares the legs that read files its code does
    not name, `# affected-reads: GLOB... [--only 'ERE']`; a declaration only adds, never narrowing
    what the text reaches: `tools/test-typecheck-scan-cases.sh` sends the scan-case tables and
    `scan_case_stubs/*` to its legs 1 and 10, its engine to every leg, so a stub edit is mapped
    there rather than UNMAPPED (gh-ocannl-1584), and `tensor/ppx_*.ml` to its `#project` legs 10
    and 11, whose `let%op` arms expand through the built ppx. A marker line outside a header stops the run, so
    a harness's fixture writes one through a variable. gh-ocannl-1797's two CI reds read this way:
    `tools/test-run.sh` reaches the members of `tools/test-test-run.sh`, `benchmarks/fixture_digest.py`
    `benchmarks/test-check-box.sh`.
  - A module of a library this build does not compile (`cuda_backend.ml` on a Mac) stays UNMAPPED,
    exit 3, but an OFF-BUILD line on stderr counts what a build compiling it reaches -- read from
    the dune files' `libraries` (a `(select)` condition included), since dune describes none of
    it here -- and names the aliases whose own code names the module: run those on a box that
    builds it, with its backend selected (9 for `cuda_backend.ml`, 667 more only link it), or run
    `affected` there for the ranked list.
  - A test or note edit gets a short, exact list: `test/operations/cost_model_template.ml` reaches
    its own alias plus the scans that read it (17 aliases in all);
    `docs/agent-notes/build-and-test.md` the 11 scans and the `doc_examples` aliases that read it, plus
    the two blind inline-test aliases: 13 (184 without the probe). One new `(test)` stanza with its
    golden in `test/operations/dune` lists that test and the scans reading the dune file, the module
    and the golden: 24 (572 without the probe).
  - A long list reaches a batch through a file, not the command line (gh-ocannl-1740):
    `tools/test-run.sh run --targets-file F [build DUNE OPTIONS...]` appends the `@` operands F
    lists, one per line, `#` comments and blank lines skipped, so `affected ... >F` then `run
    --targets-file F build -j 4` needs no shell expansion of a 75-alias list (a zsh `${(f)...}` of
    one was refused by an agent harness's safety check and cost a wrapper script). A line that is
    not one `@` operand, or a file naming none (`dune build` with no target builds the default
    alias), is refused before anything runs. `start`, `plan`, `fresh`, `repeat` (`--direct` too, whose
    operand check then reads the file's aliases) and `bisect` take the same option (gh-ocannl-1797;
    a bisect's targets may all come from the file). `tools/test-run.sh aliases PATH...` answers the
    narrower question of a test's OWN aliases, by name from `dune show aliases`, grouped
    `runtest`/`slow`/`train` as `#` comment headers -- itself a valid targets file. It exists because
    a batch built from the test files a change touched asked for `runtest-NAME` where only
    `slow-NAME` exists (`autotune_fission_sketch`, `autotune_batched_companion`,
    `autotune_mma_companion`); the ALIAS REFUSED digest now also names a refused per-test alias's
    siblings under the other families. It reads names, not the build graph: a helper module names no
    test (exit 1, pointing at `affected --files`).
  - The pre-push run is one command (gh-ocannl-1829): `tools/test-run.sh prepush --since BASE
    [--start] [--cap N] [--test-cap N] [--expect-backend B]... [build -j 4]` asks `affected --since BASE`
    for its default list (link-only aliases included), leaves off the `slow-` and `train-` members
    and runs the rest as `run --targets-file` would: same digest, verdict and exit status.
    `--start` uses `start` instead (same cap selection and fleet slot); follow a successful
    launch with `wait last` or `wait DIR`. An empty kept list exits 2: nothing started, so there
    is no run to wait for. It cannot be combined with `--list`. `train-` members run in CI's
    `@train` shards on every PR; no CI job runs `@slow`, so each `slow-` member left off is
    named on a `prepush: slow:` stderr line, for the caller to run those the change exercises
    and name them in the PR body. Before it, five workers filtered by a hand grep or a
    targets file computed inside `bash -c`, which a fleet leg's backend-pin check refuses. A leg on
    another box takes the list computed here: `prepush --since BASE --list >F` (it runs nothing;
    one alias per line), then `fleet-worker.sh execution leg ... --file F -- env OCANNL_BACKEND=B
    tools/test-run.sh run --expect-backend B --targets-file '{file}'` (ludics#864), so the runner
    stays one plain command line with its pin a word of it. It reads `affected`'s lines by an
    allowlist (one `@` operand, a `runtest-`/`slow-`/`train-` alias), and runs nothing when a line
    is anything else (exit 2), when `affected` fails (exit 2) or its verdict trailer says
    `cross-cutting` (exit 2: the full run is then CI's, above), is missing, unknown or at odds with its
    exit status (exit 2), or when it says `incomplete` (exit 3: a changed path is UNMAPPED,
    such as `cuda_backend.ml` on a Mac; `--list` still prints what it found, exit 3, to extend
    with the OFF-BUILD aliases on a box that builds them). A list with no `runtest-` member runs
    nothing, exit 0 (`--start` refuses it, exit 2), and `--list` then prints nothing, which
    `run --targets-file` refuses: check the list before shipping it.
    For a complete default CROSS-CUTTING list, the local command is
    `prepush --referenced-only --since BASE [--start] [build -j 4]`, beside tests exercising
    the change and applicable slow aliases, named in the PR body. It launches only the first
    tier (if narrow), labels saved verdicts FIRST TIER ONLY, and names slow aliases from the
    full default list, link-only ones included. It refuses a non-cross-cutting default verdict:
    otherwise `--referenced-only` is only for mid-change iteration (gh-ocannl-2072).
    Harness: leg 98 of `tools/test-test-run-options.sh`.
  - `prepush` names the harness commands the change reaches, and the one command that runs them,
    and runs none (gh-ocannl-1929): a
    change to `scripts/harness-support.sh` listed 18 aliases and no word of the shell tier, and
    its fixture break reached CI (gh-ocannl-1839). It asks `affected --harnesses --since BASE`
    (stdout is the commands; the `affected: harness:` stderr lines are not parsed, a changed path
    named `harness` printing one of that shape) and prints, as the LAST lines of stderr, a count
    line, one `test-run: prepush: harness: CMD` each and a closing line naming
    `tools/test-harnesses.sh --affected --since BASE`: after the run's verdict, after `--list`'s
    list, and after a cross-cutting or incomplete refusal or a list with nothing to run, since
    the commands are the caller's to run in each case. They change no exit status; a change
    reaching no harness prints no line (`tensor/shape.ml` beside `AGENTS.md`: none; this note
    reaches `tools/test-test-run-devices.sh`, which names it), and a failed ask runs nothing (exit 2). Named rather than run, by design: the run is ONE supervised dune
    invocation whose cap, slot kind, device probe, digest and verdict are all read from dune's
    argv and log, a harness is another program with its own verdict lines and minutes of its own
    (the six members of `tools/test-test-run.sh`), a harness of the runner starts runs of its own in fixture
    worktrees, and `--list` -- a fleet leg's form -- can only carry aliases. So a shell change's
    pre-push is `prepush`, then that closing command, and the rule to run the whole tier before
    pushing a shell change is retired to CI, which runs it on every PR. `--affected` reads
    `affected --harnesses --since BASE`'s stdout (or `--files PATH...`'s, for paths no diff
    states), never evaluates it (each line is `HARNESS` or
    `HARNESS --only 'ERE'`, HARNESS a member listed once, anything else exit 2 before the slot),
    and runs those commands in manifest order under the aggregate's one `--cpu` slot and footer;
    a group keeps that group's members, the merge build's form (gh-ocannl-1994).
    The form exists because a change reaching six harnesses meant six slot launches, or a scratch
    loop: a single harness takes no slot itself. A change reaching none exits 0 with no slot, where
    an empty group is refused: the first is a result, the second a broken manifest.
    `tools/test-test-harnesses.sh` drives the real `affected --harnesses` over a git fixture whose
    commit reaches two members, and reads the closing line's form out of `tools/test-run.sh`, so
    the two tools cannot disagree on its spelling.

## Testing a dependency candidate without changing the switch

A shared opam switch supplies every worktree on the box. Test a dependency candidate in a fresh
scratch prefix before proposing a pin (gh-ocannl-1946); do not run `opam pin`, `opam upgrade` or
`opam reinstall` on that switch for the experiment. This recipe assumes a Dune dependency whose
build dependencies are already installed in the active switch, and uses macOS-only `metal` as the
example. It checks source compatibility with the current compiler and dependencies, not an opam
solver transaction or runtime correctness. Missing prerequisites need separate coordination.

Run these commands in one Bash or zsh session; replace the source URL/ref and package name for
another dependency. Keep scratch outside all active checkouts, use absolute paths, and run each
build/install under the box's normal execution slot. The commands below are the payloads, not a
replacement for the fleet reservation or `tools/test-run.sh` wrapper.

- From the consumer checkout, set `consumer="$PWD"` and
  `scratch="$(mktemp -d "${TMPDIR:-/tmp}/ocannl-dependency.XXXXXX")"`, then
  `prefix="$scratch/prefix"`. Record `ocaml -version` and `opam var switch`; if tools are absent,
  first load the intended switch with `eval "$(opam env)"`. Use the same compiler for both builds.
- Clone the candidate with `git clone https://github.com/lukstafi/ocaml-metal.git "$scratch/dependency"`,
  then `git -C "$scratch/dependency" checkout --detach origin/main` and
  `git -C "$scratch/dependency" rev-parse HEAD`. Substitute a specific candidate ref when needed;
  record the resolved SHA, since a floating ref can move.
- Build/install without opam in a subshell:
  `(cd "$scratch/dependency" && dune build @install -j 4 && dune install --prefix "$prefix")`.
  Use a fresh prefix for each candidate so old installed files cannot survive an API removal.
- Fetch and create a detached master checkout:
  `git -C "$consumer" fetch origin` followed by
  `git -C "$consumer" worktree add --detach "$scratch/master" origin/master`.
  Here `origin` must be the staging repository (check `git remote -v`), not the issues repository.
  Record `git -C "$scratch/master" rev-parse HEAD`. A nested worktree needs its own
  `dune-workspace` root as described under Dune mechanics; this outside scratch tree avoids that trap.
- Check resolution in the master checkout:
  `(cd "$scratch/master" && OCAMLPATH="$prefix/lib${OCAMLPATH:+:${OCAMLPATH}}" ocamlfind query metal)`.
  It must print the scratch prefix's `lib/metal`, not the shared switch. The prefix goes FIRST;
  the conditional suffix preserves any existing search path without adding an empty component.
  Also inspect Dune's compilation rule with the same environment prefix:
  `(cd "$scratch/master" && OCAMLPATH="$prefix/lib${OCAMLPATH:+:${OCAMLPATH}}" dune rules _build/default/arrayjit/lib/.metal_backend.objs/byte/metal_backend.cmi)`.
  Its dependency and `-I` paths must point into the prefix, so an absent backend cannot silently
  stand in for the candidate. Adapt this target for another dependency.
- Compile master and check its dependency stand-in in the default dev profile:
  `(cd "$scratch/master" && OCAMLPATH="$prefix/lib${OCAMLPATH:+:${OCAMLPATH}}" tools/test-run.sh run build --build-dir "$scratch/master-build" --display verbose --cache=disabled @check @arrayjit/vendor_stubs/metal/runtest -j 4)`.
  The stand-in aggregate runs `runtest-metal` (the verbatim interface diff) and `runtest-runtime`
  (the runtime signature check). For another package, use its
  `@arrayjit/vendor_stubs/<package>/runtest` alias; a dependency with no stand-in has no such alias.
  A missing-package SKIPPED message is no evidence of agreement with the candidate.
  Do not use `-p` or a release profile for these OCANNL checks: OCANNL's dev-profile deprecation
  alerts are fatal, and suppressing them hides a pin that breaks master.
- Read the build's own runner log (the `log:` path in its verdict), not just the pre-check:
  `rg -F -- "$prefix/lib/metal" /absolute/path/to/run/log` must show compiler `-I` arguments
  pointing into the prefix. Verbose display, disabled artifact caching and fresh scratch build
  directories ensure compilation commands are available even if the consumer's usual `_build`
  was already up to date.
  Require exit 0 and no new deprecation alerts; do not pipe the build to `tail` or disable alerts.
  A red verbatim-interface diff means the pin must land together with the promoted stand-in copy,
  rather than proving the candidate API is incompatible. In the consumer checkout, with the SAME
  prefix still set, use
  `OCAMLPATH="$prefix/lib${OCAMLPATH:+:${OCAMLPATH}}" tools/promote.sh --build-dir "$scratch/consumer-build" arrayjit/vendor_stubs/metal/metal.mli`,
  inspect and commit the copy with the pin, then rerun the checks. A runtime signature error
  needs reconciliation of the hand-written stand-in with the candidate API, not promotion.
- Repeat the resolution check and both the dev-profile build and stand-in agreement in the
  consumer checkout:
  `(cd "$consumer" && OCAMLPATH="$prefix/lib${OCAMLPATH:+:${OCAMLPATH}}" tools/test-run.sh run build --build-dir "$scratch/consumer-build" --display verbose --cache=disabled @check @arrayjit/vendor_stubs/metal/runtest -j 4)`.
  Inspect this build's own compiler paths too. The consumer PR must build and its stand-in must
  agree; master alone misses use of a newly added API. Keep the prefix on every affected test
  invocation. Do not export it globally or persist it in shell startup files: other worktrees
  should continue using the switch unchanged.
- `@check` type-checks but does not link executables. When the dependency change affects linking
  or runtime behavior, also build the relevant executable/test alias and run the affected backend
  tests against the same prefix. Native executables need no extra loader override; for bytecode
  or a toplevel, also prefix the command with
  `CAML_LD_LIBRARY_PATH="$prefix/lib/stublibs${CAML_LD_LIBRARY_PATH:+:${CAML_LD_LIBRARY_PATH}}"`
  so it loads the candidate's C stubs instead of the switch's.
- Before any separately authorized shared-switch update, record both dependency and OCANNL SHAs,
  commands, exit statuses, the resolved package path and decisive compiler `-I` output inline in
  the PR, with log/run paths as supplementary evidence. Require both master and consumer builds
  with no new deprecation alerts, agreement checks (or an explained master diff plus the consumer's
  committed matching copy), and the change's affected tests. A successful scratch build does not
  authorize changing the switch. Leave scratch artifacts for review until the experiment is
  recorded; remove its detached worktree with `git worktree remove "$scratch/master"` only
  when it has no changes or artifacts worth preserving.

## OCaml and dune traps

- `dune build @check` type-checks; it does NOT link executables. A "rebuilt" test or benchmark exe
  after a library change is therefore the STALE binary — this has produced a false verdict three
  separate times (a timing rerun, a negative control, a guard "verified" against the old code). Build
  the thing you are about to run (`dune build <dir>/<name>.exe`, or the test's alias / `.exe.output`
  rule) and check its mtime against the sources you edited. In the other direction, a plain
  `dune build` RUNS every cram-style test executable, so it must not overlap a GPU timing window.
- A library whose name matches one of its own modules makes that module the library's INTERFACE:
  `arrayjit.context` is `(library (name context) (modules ... backends context))`, so `Context` is
  what the outside sees and `Backends`, `Schedulers`, `Cc_backend` are `Context__Backends` and
  friends — unnameable. `context.mli` therefore re-exports backend entry points honestly as
  `Context.Backends` for the raw-backend consumers that need them, and re-exports
  `module Cc_backend = Cc_backend` for the compiler-command census. Backend-independent types do
  not use that escape hatch: the footprint summary of a low-level optimized routine is
  `Ir.Low_level.footprint` (gh-ocannl-810). This also settles where a new `Context`-consuming pass
  goes: NOT a module in
  the `context` library, because `Context` would have to alias it to expose it and the alias is a
  cycle — it gets its own library on top (`arrayjit.autotune`, `arrayjit.memory_budget`). Two
  consequences when adding one. Its `.mli` should name backend-independent types through `Ir`, not
  through a `Context` re-export; an implementation that genuinely needs raw backend machinery uses
  the supported `Context.Backends` path. And a test executable's module must not
  share the new library's main module name, or it shadows the very library it tests — which is why
  `test/operations/memory_budget` became `memory_budget_planner` when `Memory_budget` moved out of
  `Context` (gh-ocannl-776).
- Most `test/operations` stanzas preprocess with `(pps ppx_here ppx_ocannl)` and nothing else, so
  `[%equal: …]` / `[%compare: …]` are unavailable — spell the comparison out (`Option.equal
  Int.equal`) rather than extending the stanza for one line.
- OCaml/Base traps that each cost a debugging session here: `Base.List.init` applies its function in
  DECREASING index order, so building a list of runs with it executes them backwards (use an explicit
  loop when the elements have effects); Base's ARRAY quantifiers — `Array.for_all`, `for_alli`,
  `exists`, `existsi`, `for_all2_exn`, `exists2_exn` — likewise walk from the LAST index down to 0
  and stop at the first decisive element: a `for_all*` stops at the HIGHEST-index failure, so a
  predicate that raises or records a reason on failure reports that one, not the first's, while an
  `exists*` stops at the highest-index success, having visited every failure above it (Base v0.17
  `array.ml`; Base's `List`
  quantifiers, `Array.find`/`findi`/`find_map` and `Stdlib.Array.for_all` go first-to-last). In
  `Fuse_epilogue`'s `covers_bijectively` (staging#1084, gh-ocannl-1293) a refusal that had to name a
  flattened component fired on the last axis's ordinary failure instead; when one failure must win,
  compute every component's verdict before judging any (as that fix does) rather than relying on
  the walk order; OCaml 5's `Lazy.is_val` returns TRUE for a lazy that is mid-force (forcing it
  then raises `Lazy.Undefined`), so it cannot guard a force reachable from inside that lazy's own
  computation; two record types sharing a field name resolve to the
  LAST-defined type, silently mistyping `x.a.b` (which is why the scheduler's event field is
  `dev_state`); and `Base.Float.max_value` is INFINITY — the finite maximum is `max_finite_value`.
- The OCaml floor (`>= 5.3.0`) runs in the scheduled matrix (the Windows CI subsection below),
  never automatically on a PR, so a change whose result depends on the compiler — a compiler
  message in a golden, a parse-tree shape a scan matches — needs the floor checked on purpose, in
  two steps. First locally, on the floor switch:
  `DUNE_BUILD_DIR=<scratch>/build53 opam exec --switch=<floor> -- tools/test-run.sh run build @<alias>`,
  where `<floor>` is the HANDLE of a 5.3 switch as `opam switch list` prints it — a global `5.3.0`,
  or the checkout's path for a local switch. The separate build directory keeps the two compilers' artifacts apart, so neither run rebuilds
  the other's tree. It does not make the run concurrent with another: `tools/test-run.sh` holds one
  run per worktree whatever the build directory, so wait for the active run (or use a second
  worktree). The switch needs the project's dependencies installed; with them, a targeted alias
  takes about a minute. Then, for evidence on the PR itself, dispatch the floor job at the head
  (gh-ocannl-1267):
  `gh workflow run ci.yml --repo lukstafi/ocannl-staging --ref <branch> -f ocaml_compiler=5.3.x -f expected_sha=<full head SHA>`.
  That runs ONE job, `Build (ubuntu-latest, 5.3.x, main)`, the scheduled floor job's full suite,
  and its check attaches to the head like the PR run's; no formatting, harness or macOS job runs.
  Its verdict is a check on the head, so a red one is a red `pr-review.sh merge` reads. When the
  floor is red on master for another cause (the scheduled sweep says so), compare the failing
  rules with the scheduled run's: if master has since fixed it, bring the base in and dispatch
  again at the new head; otherwise the merge needs `--override` naming the master red's issue.
  The guard step refuses a floor dispatch without `expected_sha`, and one combined with
  `windows_only`.
- A heap-allocation witness (`Gc.allocated_bytes` before and after the code under test) counts
  every OCaml word the domain allocates inside the window, the finalisers the GC runs there
  included. On metal those are not small: the bindings release each dead Objective-C object
  through a ctypes `msg_send` finaliser, and an executable that ran earlier routines carries tens
  of thousands of words of them. A large custom block (a bigarray) allocated in the window can
  close a major cycle and run them all, so the witness fails or passes by GC phase, and an
  unrelated merge flips it (gh-ocannl-1416: a 512-word bound read 25k words, in a different
  witness on each build). Call `Gc.full_major ()` just before the window so they run outside it,
  and print what it drained to stderr. `test/operations/broadcast_constant_init.ml`'s
  `allocation_witness` is the pattern. To find what allocates, `Gc.Memprof` sampling
  shows the call stacks, but its callbacks allocate too, so their own words land in the window.
- A scan's synthetic fixtures are parsed by the parser of the compiler RUNNING the scan, so a
  fixture spelled in syntax newer than the floor fails only on the floor leg, as a bare
  `Syntaxerr.Error` from the scanner's `parse` (5.4's `let exception E = M.E in`, gh-ocannl-1271).
  The parse tree is version-stable where the grammar is not: to cover such a construct, parse a
  floor-legal spelling and rewrite the tree (`local_rebinding` in
  `test/operations/provenance_tag_scan_cases.ml`), with the scanner taking a structure.

## What CI actually covers

- **Shards that die before dune leave every merge in that span blind to their tests.** A green
  setup or side job supplies no substitute. The span's first coverage is the next full-matrix
  `ci` run on master's tip: the daily schedule, or a dispatch on `master` with `expected_sha`
  set to that tip. A later PR's green covers only that PR's merge commit. The CI-red triage
  routine has no blind-span recovery step (gh-ocannl-2014).

- Two workflows, split by what each proves (gh-ocannl-1816). `ci.yml`, the full matrix (ubuntu
  shards, macOS, the side jobs; Windows and the OCaml floor on the schedule), runs per PR -- the
  merge gate reads the PR head's run -- daily on the schedule, and by dispatch; NEVER on a master
  push. `ci-env.yml` runs on every master push and every PR: the clean room, a fresh checkout and a
  fresh dependency switch solved from the opam files alone (no CI-side pins, no dependency cache),
  `dune build -p arrayjit,neural_nets_lib` as opam builds them, `dune build --release @check`, and
  one smoke test (`hello_world_dim1x1` on cc), also `--release`: `-p` minus its package filter, so
  the two later steps reuse the packages' objects and each module compiles once, where the dev
  profile recompiled all of them (gh-ocannl-1907; dev's warnings-as-errors stays the PR head's
  `ci.yml` gate). That is master's per-merge verdict, in minutes: each push run
  keys its concurrency group on its own commit, so none is cancelled or replaced. What the fleet's
  boxes cannot prove is exactly this, since they share one opam switch, local pins, warm `_build`
  directories and ignored fixtures; what the hosted full matrix would add on a merged commit
  (beyond the PR head's run) the fleet's integration loops and the daily sweep cover. Readers of master's
  verdict outside this repository (ship-pr's base checker, the fleet's launch gate, the CI-red
  triage routine's prompt) have no `ci` push run to read: they read ci-env's runs, the schedule's,
  and the execution registry's integration records.
- `dune build @bin-smoke` runs every `bin/` executable sequentially on deliberately tiny workloads
  with the `cc` backend pinned (gh-ocannl-858), plus one macOS-only contribution outside `bin/`:
  `benchmarks/runners/ocannl/metal_queue_probe.exe --iterations=1024`, the exact-loop-count mode
  that takes milliseconds and exercises the whole Metal path — library compile, the three
  command-buffer submission shapes, SharedEvent signal/wait — so a binding change that breaks it at
  runtime fails here rather than the next time somebody reruns the probe by hand (gh-ocannl-905).
  Exact iterations means no calibration and so no timing tolerance to lose on the contended macOS
  runner; the arm ratios it prints are not asserted on. That rule also sits on `@metal-codegen`,
  since it names `metal` in its backend marker. It is an exit-status canary only: its output and
  timings are not goldens and make no measurement-validity claim. Per-PR CI includes it in the main
  `@default @runtest` Dune walk, so `@check`'s inability to link or execute an executable no longer
  leaves first-iteration failures in these tools uncovered. Keep benchmark-reproducibility work in
  gh-ocannl-743 rather than expanding this alias into a benchmark assertion suite.
  `bin_smoke_membership_scan` (gh-ocannl-874) derives the public executable declarations from
  every Dune file below `bin/` and every recursive-alias contribution from the repository's dune
  files, then requires exact one-to-one membership; its separate negative-control rule runs the
  same checker on a synthetic omitted member and accepts only the failing exit status. The scan
  preserves repeated command sites so a duplicated smoke is not collapsed into a set, follows
  `(alias ...)` dependencies transitively, and exempts exactly two things: the no-argument
  `env_spelling_gate.exe` invocation on the infrastructure alias that owns it, and the
  platform-gated probe pinned in `platform_contributions` — by dune file, program path and the
  literal `(= %{system} macosx)` condition, and only while the stanza runs that one program, so an
  unconditional run, a different condition, a second command beside it, or the same stanza in
  another file is refused as before (a public executable whose only smoke run is platform-gated is
  still reported missing). Every
  command-bearing alias contribution, root or transitive helper, must depend directly on
  `(universe)`, so Dune cannot cache away runtime coverage after another contribution reruns.
  Alias edges and generated file producers in a public executable's `link_deps` or
  `preprocessor_deps` are also traversed and refused if they run a public executable or contain an
  opaque dependency/action: they execute while the smoke target is being built, before its credited
  runtime canary.
  It fails closed on other private helpers, external or otherwise opaque actions, `dynamic-run`,
  `with-accepted-exit-codes`, `enabled_if` on a public
  declaration, `alias_rec`, implicit built-in aliases, implicit test runners on arbitrary aliases,
  `data_only_dirs`, dependency-list `include`, `read*` expansion or an otherwise unmodeled pform,
  explicit (file or directory, including files below a directory target) or
  action-inferred generated-target dependencies (including dependency pforms in fields or actions,
  pforms embedded in larger dependency/action atoms, literal file-input action positions, and
  output actions under a literal `chdir`, including `mkdir`'s directory kind), target-bearing alias
  rules, a pform in an inferred output path, an unresolved `chdir` around inferred output targets,
  an action preprocessor on a public executable or any library, an explicit `install` into section
  `bin`, or unexpanded top-level `include` stanzas. A bare executable name is itself refused: its
  `.exe` suffix does not stop ambient PATH from selecting a program outside the workspace.
  Action-local or directory-level PATH rewrites (under any case
  spelling), and an `env` stanza's `binaries` mapping, are opaque too, including in transitively
  reached aliases. A
  directory-level override follows Dune's scope: it reaches descendants of its own `(subdir …)`
  placement, not the parent or a sibling. Paths expanded from `%{exe:…}`, `%{dep:…}`, and named
  dependency pforms remain anchored to the stanza when the process runs under `chdir`; literal
  program paths remain relative to the changed working directory. Each construct needs deliberate
  scanner support before it can participate in this static guarantee. Absolute executable paths are
  refused before normalization, so `/../bin/tool.exe` cannot collapse into a workspace identity.
  A target-producing rule anywhere in the repository may not run a public bin executable directly:
  its output could be an implicit build prerequisite of a smoked executable, hiding an extra
  execution outside the alias dependency edges the census can see. For rules that produce a
  declared public executable's generated source or interface below `bin/`, the same check
  recursively covers alias dependencies and
  refuses shell, interpreter, private-workspace-generator, or otherwise opaque commands plus
  unexpandable dependency specifications. Producer-side executions are errors rather than smoke
  credit because an unrelated generated target may never be built. Commands under absolute
  `chdir` destinations are likewise refused before path normalization can turn a host path into an
  apparent workspace identity.
- GitHub CI exercises exactly ONE OCANNL backend. `test/config/ocannl_config` pins `backend=cc`,
  and no runner has a CUDA or HIP device, so a green `ci` run says nothing about the Metal, CUDA or
  HIP *backends*. Do not read a green PR check as cross-backend validation; it is a CPU-backend and
  portability check. The macOS runners are not deviceless, though, and reading this bullet as "no
  Metal at all" is what cost a review round: `MTLCreateSystemDefaultDevice` returns an `Apple
  Paravirtual device` there, and `@bin-smoke`'s `metal_queue_probe` run uses it — library compile,
  three command-buffer submission shapes and SharedEvent signal/wait, about 50 ms on every macOS
  `main` leg (gh-ocannl-905). That is a canary for the Metal BINDINGS; OCANNL's Metal backend, its
  codegen and its scheduling are reached by no CI leg, and that is a decision (gh-ocannl-942,
  closed): Metal is the development Macs' everyday backend and a leg of `tools/sweep.sh`, while a
  hosted leg on the paravirtual device would have to skip simdgroup matrix work and would inherit
  the macOS runners' contention — runner time for a subset of what the reference Mac already runs.
- The per-PR macOS leg is advisory at merge (`.github/ship-pr-advisory-checks`,
  lukstafi/ludics-lite#720): `pr-review.sh merge` does not wait for it, and the wave gate does not
  hold behind an older merge's run going only on it. Every test it runs, the ubuntu shards run too,
  so its unique coverage is the platform (arm64, Apple clang, the Metal queue probe above). In a
  ~20-PR wave its queue held merges and launches for about two hours, while the reds it alone
  would catch are rare. A macOS-only red therefore lands on master, where the scheduled sweep's
  macOS legs find it and the CI-red triage routine owns it. A change that is macOS-specific runs its suites
  on a development Mac before merging (or waits for the PR's own macOS leg on purpose).
- The benchmark tests' torch legs run in ONE job, `torch-runner`. The main legs run
  `benchmarks/test_orchestrate.py` through dune on runners without torch, where the tests that
  drive `benchmarks/runners/pytorch/run.py` for real skip. `torch-runner` (ubuntu, no OCaml
  toolchain) builds a CPU-only venv from `benchmarks/requirements-ci-torch.txt`, cached on that
  file and the exact Python, and runs the whole file with `BENCH_TORCH_REQUIRED=1`, under which a
  torch skip is a failure. A new torch-dependent test therefore gets that job's coverage by taking
  its venv through `torch_venv_or_skip`, with no list to extend; a test that probes torch its own
  way skips there too, unseen. Bumping a pin in the requirements file rebuilds the cached venv.
  Tinygrad is not provisioned: its runner has no CI leg.
- A red on merged master is presumptively CLAIMED work. `ci.yml`'s `notify-triage-routine` job
  fires the "ocannl-staging CI-red triage" Claude Code cloud routine on any non-PR master red —
  the scheduled sweep and plain dispatches — and `ci-env.yml`'s job of the same name on a red
  master push (a logged no-op until the `ROUTINE_FIRE_URL`/`ROUTINE_FIRE_TOKEN`
  repo secrets are set; a lost fire is caught by the routine's own daily backstop sweep). The
  routine claims the red with an issue on **ahrefs/ocannl** titled
  `CI red on master@<short-sha>: <workflow>` — issues are disabled on the staging repository, and
  ahrefs/ocannl is where the issues live anyway (AGENTS.md's two-repository rule) — and either
  opens a `ci-fix/*` PR on staging (it never merges its own PRs) or posts its diagnosis to that
  issue.
  A ci-env red names its merge (one run per push). A full-matrix red on master comes from the
  schedule and covers a SPAN, about a day of merges.
  Narrow a red span of more than one merge by hand: dispatch `ci` on `master` with `expected_sha`
  set to a commit inside it, bisecting, down to the floor its local actions set
  ([How far back a dispatch reaches](#how-far-back-a-dispatch-reaches)). On master an ancestor of the head is accepted, the run is
  named `ci at <sha>`, and its red fires no triage. A red only the OCaml-floor job shows bisects the
  same way with `ocaml_compiler: 5.3.x` added, which runs that one job instead of the ordinary
  matrix (which never builds the floor).
  Merging sessions do not watch CI after landing (roll-forward, ahrefs/ocannl#861); before fixing
  a master red by hand, find the claiming issue and any linked PR, and take over only where triage
  visibly stopped short — saying so on the issue first.
  In a red `dune build` log, the failing rule is the one under dune's `File "<dir>/dune", line N`
  header (grep the job log for `File "`), never simply the last test to print: dune keeps running
  the other rules after a failure, and a test's chatty stderr can land last on a step whose only
  red is far earlier (gh-ocannl-1271).
- The first shard of the Ubuntu/OCaml 5.5 main suite preprocesses `cc_backend.ml` with
  `OCANNL_LOG_LEVEL_CC_BACKEND=3` and builds both `@check` and
  `@test/operations/runtest-cc_backend_trace_name` in that environment. The focused runtime test
  executes one cc routine and checks the bare result inside its `work` trace against the compiled
  routine name: `@check` alone only type-checks, so a trace expression that silently resolved a
  different in-scope binding would otherwise remain green (gh-ocannl-859). The test reports an
  explicit skip in ordinary level-0 suite runs, keeping the extra execution confined to this gate.
- `cuda_backend.ml`, `hip_backend.ml` and `metal_backend.ml` each live in an `(optional)` library
  over `cudajit`/`hipjit`/`metal`, and `arrayjit.context` and the vendor-gated test and tool
  executables reach their vendor code through a dune `select` whose fallback arm is a
  `*.missing.ml` stub; with the vendor package absent both mechanisms succeed SILENTLY, so an exit
  status cannot distinguish "compiled" from "skipped". `arrayjit/vendor_stubs/` (gh-ocannl-1652,
  gh-ocannl-1290, gh-ocannl-1688) closes the TYPE-CHECKING half of that gap everywhere: it compiles
  copies of the three backends, of `context`'s `select` vendor arms, and (`arms/`) of every test
  and tool executable's vendor arm against interface-only stand-ins for the three packages, so
  `dune build @check` on any machine (per-PR CI among them) fails on a type error in any of them or
  on an `arrayjit` interface change one of them depends on (a value hidden from `cuda_mma.mli`, a
  changed `Backend_intf` signature). The stand-ins are verbatim copies of the installed packages'
  `.mli` files, and `@arrayjit/vendor_stubs/<package>/runtest-<module>` (`cuda`, `nvrtc`, `hip`,
  `hiprtc`, `metal`) diffs each against the installed interface where the package is installed
  (part of `runtest` there, so the daily sweep runs it on rog, minix and tuf, and every macOS build
  runs `metal`'s), prints `SKIPPED` elsewhere, and is regenerated by `dune promote` on the box after
  a package upgrade; arrayjit pins `metal.dev` to ocaml-metal's `main`, so an interface change
  pushed there reddens `runtest-metal` on the next macOS build until the copy is promoted. The one
  stand-in with no interface to copy, `metal/runtime.mli` (`metal.runtime` infers its signature,
  and its `Objc` includes all of Ctypes), is a hand-written subset that `runtest-runtime` checks by
  compiling `module _ : <its text> = Runtime` against the installed library: a signature check, not
  a diff, so it catches a promise the real library breaks but cannot regenerate itself. On
  macOS `metal` is the real package, so what the stand-in catches off the Mac is shown by Linux
  CI, not by a local build. The tree sits outside `test/` deliberately:
  its copied backend sources are library code, and the test-source scans would read them as tests;
  `env_var_deps` reads the backend copies but not `arms/`, whose sources it already reads at their
  own `select`. An arm compiled as a library module is checked slightly more permissively than in
  its executable (an unused top-level value is an error only under the executable's empty
  interface). A stub is a check of signatures only: it proves nothing about the vendor library's
  behavior, so a change to a backend's runtime behavior still needs its box. A NEW vendor-gated
  `select` arm that no library under `arrayjit/vendor_stubs/` copies and compiles is refused by
  `test/operations/vendor_stub_arms_scan` (a `@scans` member), naming its dune file: copy it into
  `arms/dune` as its siblings are. The vendor libraries are derived from the stand-in tree.
  Verify on the box carrying the toolchain — rog for cudajit, minix for hipjit, through whichever
  boot is up (`rog-nv-linux`/`minix-amd-linux` on native Ubuntu, `rog-nv-wsl`/`minix-amd-wsl` on
  WSL; the site's `kind_of` says which) —
  and check two things there rather than one: that
  `_build/default/arrayjit/lib/.<backend>_backend.objs/byte/<backend>_backend.cmi` exists, and that
  the `select` landed on the vendor arm, which
  `head -1 _build/default/arrayjit/lib/<backend>_backend_impl.ml` names outright — dune copies the
  chosen arm under a `# 1 "…"` line directive naming it, and `.missing.ml` there is the stub. Each
  box carries exactly one of the two toolchains, so the OTHER backend's absent `.objs` on that same
  build is a free negative control. `@check` also proves compilation and never execution, so pair it
  with a runnable probe wherever one exists. Two PRs in two days paid for this: gh-ocannl-758
  (staging#490) shipped a HIP arm unparsed beyond syntax and edited the CUDA arm blind the next day, and
  gh-ocannl-773 (staging#494) touched both again. `tools/machine-verify.sh` (gh-ocannl-796, renamed
  from `tools/remote-verify.sh` by gh-ocannl-1047) scripts that off-box loop, and it stays the
  pre-merge check for anything beyond type-checking. Installing cudajit or hipjit on a runner means a
  CUDA or ROCm toolkit per run (gh-ocannl-794, closed), which is why the stand-ins are interface
  copies the agreement rule keeps honest rather than a hand-kept second API; the daily sweep
  compiles and runs both backends on their boxes, so a runtime break that skipped machine
  verification surfaces on master within a day, under the roll-forward policy.
  Metal is the same shape -- an optional library over the `metal` package behind a `select` -- and
  `--expect-lib metal` proves it on mac-studio.
- `tools/machine-verify.sh BOX BRANCH` is the one-off counterpart to the scheduled sweep for a pushed
  branch. `tools/remote-verify.sh` survives as a forwarding shim that prints a deprecation line on
  stderr; new callers name the new script. The verified machine's procedure is ONE file,
  `tools/machine-verify-far.sh` (POSIX sh: dash on Ubuntu, the macOS `/bin/sh` on mac-studio), fed
  unchanged through the same quoted argument string either to `ssh BOX` or, when BOX is this
  machine, to a local `/bin/sh -c` under `env -i` from `$HOME`, keeping only the session basics
  (HOME, USER, LOGNAME, PATH, SHELL, TMPDIR, LANG, LC_ALL, LC_CTYPE, SSH_AUTH_SOCK) -- so a check
  added to the procedure runs on both transports, and a caller's `OPAMSWITCH` or Dune settings reach
  it no more than they reach an SSH session. The local transport exists because self-ssh fails on
  the native fleet boxes (no `known_hosts` entry for itself, no key authorized for self-login), which
  left a worker resident on a GPU box unable to verify there (gh-ocannl-1047). Placement asks what
  `ssh BOX` would reach, not what the host calls itself: Apple names mac-studio `LukaszsacStudio`,
  and a table of the site's box names here would be a second copy of the ssh configuration. `ssh -G
  BOX` names the endpoint offline, and BOX is this machine when every address that endpoint
  resolves to can be `bind`-ed here (the kernel's own "is this address mine", the same on Linux and
  macOS with no `ip`/`ifconfig` to parse; Debian's `/etc/hosts` maps the box's own name to
  127.0.1.1, which Linux binds), at port 22 and with no ProxyJump/ProxyCommand. No local address
  means SSH; a split answer (some addresses here, or a local one on another port -- a forward into a
  VM) is refused as ambiguous rather than guessed. `--local` makes "BOX is here" an assertion:
  it refuses, naming both hosts, when no endpoint address is this machine's, so a wrong box name
  can never run silently on the wrong hardware; `--ssh` forces the old transport; `localhost`
  always runs here. The whole-trip cap (`--trip-cap`, formerly `--ssh-cap`, still accepted) and
  the per-command cap are the same perl process-group supervisor on both transports -- never
  `timeout(1)`, which macOS lacks. BRANCH may instead be a full 40-hex commit that a branch of the
  staging remote contains, so measuring a merged commit or a PR head pushes no throwaway branch
  (gh-ocannl-719 pushed two `claude/issue-719-measure-<sha>` for it). The object being present in
  BOX's checkout proves nothing -- it may be an unpushed local commit, or one the remote holds
  unreachable -- so the procedure fetches the remote's branch heads now, into a private
  `refs/machine-verify/heads-<pid>` namespace (the checkout's remote-tracking refs never move),
  requires one to contain the commit, names it in the provenance (`reachable from:`, preferring
  master) and empties the namespace -- before the fetch too, since a verifier killed past its cleanup
  leaves one behind that a later run drawing the same PID would otherwise read a deleted branch from. For a sweep inside that worktree,
  `BENCH_RESULTS_DIR` points `benchmarks/orchestrate.py`'s `results/` outside it, and
  `OCANNL_AUTOTUNE_CACHE_DIR` the autotune cache, so nothing needs copying out before the worktree
  is removed. The procedure derives the remote pointing to the staging
  repository by URL, fetches the named branch, without rewriting the checkout's `FETCH_HEAD`, resolves one commit, creates a fresh detached
  worktree, resolves the checkout's selected opam switch before leaving it, runs explicitly under
  that switch, and removes just that worktree before its exit sentinel (never repository-wide
  `worktree prune`, which could unregister an unrelated temporarily unavailable worktree). The individual
  commands (including worktree add/remove) and the whole trip have separate process-group caps; the latter also bounds setup
  and cleanup, and on the local transport its signal still lets the procedure remove its worktree.
  Non-login shells receive the CUDA/WSL PATH prefix that `tools/sweep.sh` uses.
  Ambient `OCANNL_*` variable names are printed and cleared before opam runs; names injected by the
  selected switch are printed and stripped inside `opam exec`, so only the requested backend can
  override the pushed tree's configuration. A regular, non-symlink root `ocannl_config` from the pushed commit
  is the configuration boundary; when the commit has none, the script creates an empty ignored one
  in the disposable worktree so a personal file above it cannot reach root-launched probes. A
  worktree root nested under any outer Dune root is refused: without that
  boundary Dune can build the parent checkout while this script reports the detached commit.
  `--expect-lib cudajit|hipjit|metal` asserts all three
  pieces of optional-backend provenance above (positive `.cmi`, vendor `select` arm, and the other
  two GPU backends' absent `.cmi` -- each fleet box carries one vendor package), then runs
  `bin/device_props` under the pinned backend and puts the tile-MMA capability it reports on the
  verdict line as its `tile_mma` field, a tile such as `16x16x16` or `none` (gh-ocannl-1070). A compiled, selected
  hipjit said nothing about what it EMITS: with `HIP_PATH` unset -- which is every hermetic leg --
  the backend searched no distro `/usr` tree, found no rocWMMA headers, and every WMMA test ran the
  scalar fallback green. For hipjit the capability is asserted from two halves no box list
  restates: the backend's own per-device `tile_mma_eligible` (its gate's device predicate, surfaced
  in HIP's `static_properties`) and a header probe kept independent of the backend's search, the
  tree `hipconfig --path` names holding `rocwmma.hpp` and `rocwmma/internal/types.hpp`. Both present
  and no capability fails the trip; either absent prints `NONE -- ... scalar fallback` with the
  reason and passes, since that box cannot tensorize. A test, probe or
  `--record-golden` trip also requires a backend and
  asserts `_build/default/test/config/ocannl_backend.txt`; an unrestricted test alias is reported
  only as passing under that configuration, since the alias may be backend-independent, while a
  runnable probe must print its own backend/device evidence. An `@check`-only trip says explicitly
  that it compiled code and executed no backend. The usage header says the backend requirement
  where the operations are listed, with a `--backend cc --test` example: written from a header
  that listed `--test` as free-standing, a worker's first trip was refused before any ssh and cost
  a reservation round-trip (gh-ocannl-986). The dune width defaults to BOX's own, not a flat 4
  (gh-ocannl-986): the per-slot cap `tools/test-run.sh` injects into a batch there (bullets
  below), so a GPU leg a worker runs without `-j` stays within the width the box's correctness
  slots were measured at. It is decided on the far side, where the devices are, by the CALLER's
  `tools/box-jobs.sh` and `tools/batch-backends.sh` -- passed as arguments like the supervisor
  source, so a verified commit older than them cannot change them -- evaluated in a bash child
  (they are bash; the procedure is POSIX sh) after the ambient `OCANNL_*` clearing, so it probes
  the real devices. Like `test-run.sh`, it judges every backend the trip can HOLD, not just the
  pinned one: `batch_resolve` over the trip's aliases in the pushed worktree (a reached stanza
  marked `; ocannl-backend: cuda` holds that GPU under `--backend cc`), every backend for a
  `--run` probe or an unread answer (a tree predating the readers), and the tightest cap wins.
  Where no backend meets a cap (CPU backends off rog, metal, a box without a GPU) nothing is built
  and the width is 4; an explicit `-j` always wins; the provenance's `dune jobs:` line and its
  `machine-verify: batch:` lines say which it was and why. A `--run` probe gets the width as
  `DUNE_JOBS`, so a bare `dune` inside it is capped too, while a `-j` of its own still wins.
  Dune reads `DUNE_JOBS` only from 3.22.0 (below the project's 3.20 floor the probe line says the
  variable is ignored), and `DUNE_CONFIG__JOBS` is not a setting at all: timed on 3.24.2, two
  2-second rules took 4.1 s under `DUNE_JOBS=1` and 2.05 s under `DUNE_CONFIG__JOBS=1`. Golden mode prints the corrected `.actual`
  contents and an apply-ready patch, then re-runs the alias before accepting it so a second failing
  dependency cannot hide behind a promotable diff. Before reset, source status (with untracked-file
  reporting forced independently of Git configuration) must name exactly the listed golden
  destinations, both after promotion and after the re-run. After reporting it, the script resets tracked
  files, removes untracked files, and proves the worktree clean at the resolved commit before
  running another operation. Every path also reasserts exact HEAD, clean tracked/untracked source,
  and the unchanged configuration boundary after each operation and before the final certificate;
  a nominally successful probe that edits its checkout therefore fails loudly. Do not replace its
  unpiped output with a convenience pipe: the procedure travels on a separate file descriptor
  while child stdin is `/dev/null`, the procedure's `exit:` sentinel is the build verdict plus
  cleanup, and the caller's `ssh exit:` / `local exit:` sentinel is the transport's verdict.
  `tools/test-machine-verify.sh` runs the shipped pair and the shim through fake SSH/opam/Dune and
  real disposable Git repositories, directly in Ubuntu CI rather than inside Dune; the fake `ssh -G`
  places BOX at 192.0.2.1 (TEST-NET-1, never assigned) or 127.0.0.1, and the fakes read their
  controls from a file because the local transport clears the environment. It checks actual
  checkout/environment observations, failure reasons, timeout statuses, which transport carried the
  trip (observed at the fake ssh), placement refusals, cleanup ownership and golden restoration;
  source-assertion, commit-containment, namespace-clearing, golden-scope, local-address and
  environment-clearing mutants must fail the same oracles. The width legs run the subjects from a
  scratch copy of `tools/` whose `box-jobs.sh` has one line appended, pointing the table's
  `OCANNL_TOOL_*` device probes and `FLEET_LOCAL_BOX` at the case's control file (the far side
  clears every ambient `OCANNL_*`, as it must), and at stand-ins for the two backend readers;
  each observes every build's `-j` at the fake dune, and mutants restoring the flat default or
  judging the pinned backend alone must fail them.
- Besides the compiler (next entry), the cc VECTOR WIDTH separates an ARM box's pass from the
  ubuntu leg's (16 bytes against 32 or 64; `macos-latest` is ARM too, so per PR only ubuntu sees
  it). The local proxy is a targeted rerun at `OCANNL_CC_VECTOR_BYTES=32` in a stanza that declares it; when and
  why: [backend-precision-and-simd.md](backend-precision-and-simd.md#simd-widths-and-narrow-storage-performance).
- `tools/ci-compiler-test.sh` is the cheap local proxy for a compiler-sensitive Ubuntu CI failure
  (gh-ocannl-846): it downloads the GCC 13 packages with `apt-get download`, extracts them into a
  scratch prefix with `dpkg-deb -x`, and runs exactly one named `runtest-` alias in a fresh Dune
  build directory with `OCANNL_CC_BACKEND_COMPILER_COMMAND` pointing at a logging wrapper. A pass
  requires that wrapper to have been invoked, so an irrelevant alias or a cached action cannot be
  certified; the run goes through `tools/test-run.sh` with a configurable cap. Ambient OCANNL
  configuration, generic compiler/header selectors, and the full `LD_*` dynamic-loader override
  family are cleared before the explicit settings are
  injected, while the harness-control `OCANNL_TOOL_*` namespace is preserved. `--aarch64-clang`
  additionally creates an isolated scratch apt index for the host's `VERSION_CODENAME` at
  apt.llvm.org (with the signing key's SHA-256 pinned in the script), then unpacks clang 21 and
  arm64 cross headers, derives `LD_LIBRARY_PATH` from the unpacked `libLLVM.so.21`, and sets
  `AARCH64_CROSS_GCC` to a second logging wrapper using `--target=aarch64-linux-gnu` and Apple's
  NEON assembly dialect, with the Debian cross-header directory passed explicitly through
  `-isystem`; today `cc_march_census`'s `foreign` pieces
  (`runtest-cc_march_census-w16_native_foreign` and its siblings) are the tests that consume that
  hook. The real
  download is deliberately x86_64 Linux/Debian-family-only, since its cc kernels execute on the
  host; `--dry-run` validates and prints the complete staging plan on macOS and other hosts. The
  isolated sources use `signed-by` (LLVM names the digest-checked key), with empty
  `Dir::Etc::trusted` and scratch `trustedparts`; `ci-aarch64-cross.sh` uses the same trust
  isolation. `Dir::Etc::trusted=/dev/null` draws apt 3.2's deprecation warning whenever a
  source has no `signed-by`; leave it empty. This is
  compiler/codegen evidence, not an OS emulator: the GCC 13 packages come from an isolated scratch index over Ubuntu 24.04 (noble), the series
  `ubuntu-latest` runs, so a newer host's own `gcc-13` (resolute's 13.4 on minix) cannot stand in
  for CI's 13.3 (gh-ocannl-1696; the exact version and target are printed and major 13 is
  enforced). `--toolchain DIR` keeps that GCC in DIR, staged on first use and reused unfetched
  afterwards: minix keeps one at `/home/lukstafi/.local/opt/x86_64-linux-gnu-gcc-13`, the census
  leg is one invocation, `tools/ci-compiler-test.sh --toolchain <dir> --pieces census`, which runs
  every `cc_march_census` piece in turn (the list read from `test/operations/dune`'s rules) and
  prints one verdict per piece, each held to its own exit status and its own calls to the fetched
  GCC (gh-ocannl-1852), and a fresh-build-directory run costs
  seconds to minutes once dune's cache is warm. Its isolated `apt-get update`s are capped and
  retried as CI's are (gh-ocannl-1809, the workflows' apt entry below): a stalled mirror ends the run in
  about 9 minutes, naming apt, rather than hanging the leg. Run it under the fleet's slot with `--cpu`. The
  clang leg has a Linux cross sysroot rather
  than the macOS SDK, ABI, linker or runtime. It therefore complements CI and the GPU boxes rather
  than replacing either. Fetches, extraction and the test harness are attached children: an outer
  cancellation forwards `TERM` and reaps the active child before scratch cleanup.
- `tools/ci-durations.sh` is the source for revisiting the `timeout-minutes` ceilings in
  `ci.yml`: it aggregates the last N completed runs of a workflow (`--repo`, `--workflow`,
  `--branch`, `--event`, `-n`) into one row per (job name, conclusion) with count/min/median/max
  minutes, derived from each job's `started_at`/`completed_at`. Do not reach for
  `runs/<id>/timing` instead — that endpoint reports zero billable milliseconds on every job
  here. Filter by `--event schedule` for the extended (Windows, 5.3 floor) matrix, whose jobs are
  the ones with the widest spread. `tools/ci-times.sh` answers the neighbouring question for a
  SINGLE run: where its minutes went, step by step.
- **A job GitHub never ran still carries timestamps, and they run BACKWARDS.** A skipped job's
  `completed_at` precedes its `started_at` — by one second in the case measured (gh-ocannl-901 on
  staging#611), which the plain subtraction rendered as `-1m59s`, since Python's `divmod(-1, 60)`
  floors to `(-1, 59)`. So a reader saw a two-minute job that never started. Both readers of these
  pairs now treat a negative interval as no measurement rather than a small one: `ci-durations.sh`
  skips the job and counts it on stderr, `ci-times.sh` prints `(no time)`. That label also covers a
  completed job with an absent endpoint; a job that is merely still `queued` or `in_progress` keeps
  its status as the label, since there its duration is missing only YET. `tools/test-ci-times.sh`
  is the hermetic harness — a fake `gh` serving one recorded `runs/<id>/jobs` payload that mixes
  measurable, absent and reversed pairs, with mutation twins that restore each dropped guard and
  must be rejected for printing `-1m59s` and `(completed)` respectively. Like its siblings it is on
  no dune alias and rides the Ubuntu shell-harness CI step (the `group_alive` bullet above has the
  SKIP contract they share), here because the subject invokes `python3` under that exact name,
  which is not what every platform `dune runtest` covers calls it — so a host without one skips
  every leg rather than deciding none of them quietly.
- Both timing tools use `tools/ci-timing.py` for interval parsing: missing, invalid and reversed
  timestamps are unavailable, equal timestamps remain zero. Cross-run `tools/ci-durations.sh`
  accepts `--job REGEX` and `--step REGEX` (Python regex search); the latter sums matching steps
  per job and reports step minutes plus paired job/rest minutes and per-job share percentages.
  Select renamed steps with an alternation; matched names and missing/unusable/unpaired counts
  stay visible. Skipped alternatives are excluded before interval completeness; all-skipped
  selections stay unavailable. Cancelled/failed jobs with usable times remain
  separate conclusion groups. A share/rest sample needs a positive job interval and a step sum
  no larger than it. Medians summarize per-job shares, not a ratio of aggregate medians.
  `tools/test-ci-durations.sh` runs synthetic cross-run API fixtures in the shell-harness CI step.
- The regular suite does not run the training integrations. `bigram`, `mlp_names`, `mlp_bn_names`,
  `circles_conv`, `fsm_transformer` and `transformer_names` sit on the `train` alias — a third
  tier beside `runtest` and `slow`, for runs that are toy-sized by intent but serialized on the
  `ocannl_training_test` lock, which made them CI's wall-clock tail on every substrate measured
  (2026-08: ~95s/195s/246s cc for the last three on a 24-thread Linux box; the ubuntu runner's
  whole `dune runtest` step was ~8min, most of it this chain; taking it off the per-PR path took
  ubuntu CI from ~31min to ~15 and macOS from ~16 to ~10). The daily sweep's full-suite units
  build `@runtest @train` together, so every backend still executes them daily, and per-PR CI
  runs the tier as two ubuntu shards (gh-ocannl-1422; the schedule adds a macOS `train` job). It
  used to be one macOS job, because one ubuntu job would have set the matrix's latency (the chain
  is ~10min on a 24-thread Linux box with the compile cached, slower on the 4-vCPU runner); but
  a second macOS job per PR doubled each PR's claim on the scarce macOS runners, and in a ~20-PR
  wave macOS jobs queued ~2h. `tools/ci-shard.sh targets K/N train` deals the `train` aggregate's
  members by `TRAIN_WEIGHTS` -- WALL seconds, since the members serialize on the lock and each
  saturates the cores, so a shard's time is the sum of its members' -- re-measured with
  `tools/test-run.sh fresh @test/training/train-<name>...` on a Linux box, which prints each
  member's wall time. A change to training dynamics, `Train.*` plumbing or the autotuner's
  fission path is still worth a local `dune build @train` (or the affected
  `@test/training/train-<name>`) before pushing, since macOS-specific breakage there surfaces
  only in the schedule and the sweep. `@slow` keeps the
  genuinely long real-dataset runs (`mnist_conv`, `cifar_conv`, `gpt2_dry_run`). CI also runs its
  one dune walk as one `dune build @default @runtest @bin-smoke` — two separate commands would
  hold the serialized lock chain (what remains of it) against an otherwise idle runner after every
  file target finished. The targets live in the job's `SUITE_TARGETS`, which the Windows leg's
  PowerShell step splits into an array: PowerShell splats an unquoted `@word` on a command line to
  nothing, a plain `dune build` that runs no tests.
- Per-PR ubuntu runs that suite as N jobs (`shard: K/N` in `ci.yml`'s matrix), macOS as one: as a
  single job ubuntu had grown back to 18-32min, three quarters of its dune step executing
  `test/operations`, and macOS runners are the scarce ones. `tools/ci-shard.sh targets K/N` (suite
  `main`, the default; `train` above is the other) names
  a shard's targets: each `runtest-<name>` alias of `test/operations` (built `@@`, non-recursive)
  is a unit, and one `@rest` unit is every suite alias, non-recursively, of every OTHER directory
  holding a tracked dune file. **`@@dir/default` is not non-recursive**: no dune file here defines
  `default`, and dune's implicit one is `(alias_rec all)`, so `@@./default` builds the whole tree's
  `all` — and `all` RUNS every `(test)`, whose stdout capture is a file target, so one shard
  would execute nearly every test. A directory's share of `@default` is `@@dir/all`, and the
  script refuses a dune file that defines `default` itself. The units cover the suite because a directory's `runtest` is its
  per-test aliases (dune generates one per `(test)` and inline-test library; `env_var_deps` holds
  every hand-written one to the `runtest-<name>` shape and the aggregation) and its `all` adds
  only what they already build — checked when the split landed by running both shards in fresh
  build directories under `--trace-file` and comparing their executed commands with an unsharded
  run's; repeat that check after adding an `all`-only target to `test/operations`. The units are
  dealt longest-first by the script's `WEIGHTS` table (the costs are heavy-tailed, so an
  unweighted deal measured 63/37). Its unit is CPU seconds (rusage user + system, a test's
  compiler children included) ON THE RUNNER (gh-ocannl-1503), because they add up: a shard's
  tests keep the 4-vCPU runner's cores busy at `-j 4`, so its dune step lasts about a quarter
  of its CPU. Wall-clock seconds of job slot (gh-ocannl-1464's unit) do not add up: a test's wall-clock is
  its CPU stretched by its neighbours, worst between tests running OpenMP kernels, whose libgomp
  barrier spin is sized for their own threads, not the runner's. **The build jobs run with
  `OMP_WAIT_POLICY=passive`** for that reason (gh-ocannl-1544): built side by side with and without
  it in three dispatch runs, the suite went from 125 to 79 CPU-minutes and main 1/2's dune step on
  one CPU model from ~22 to ~13.5 minutes, every golden unchanged; the autotune tests, which repeat
  small parallel kernels, moved most (autotune_sketch_placements 328 -> 50 s). An A/B of a CI
  setting runs as such a pair: a throwaway branch whose `ci.yml` lists only the jobs under test,
  each twice by an extra matrix key, dispatched with `expected_sha`, so both arms draw runners at the
  same time -- and compare jobs on the same CPU model (the `host` line's `cpu=`), which moves a
  shard's time by half (EPYC 7763 against 9V45). A deal balanced on wall-clock
  weights measured under the previous deal predicted 60.5/60.5 minutes of slot and ran 80.6 on one
  shard while the other hit the job's 45-minute timeout. A Mac-measured table drifts as badly
  (gh-ocannl-1464 saw 26/12): measure on the runner. A single unit whose own wall-clock outlasts
  the rest of its shard is the one cost CPU hides, so CI also prints `ci-shard-wall` lines and
  `plan` names each shard's longest unit by the script's `WALL_WEIGHTS` (`merge-weights --wall`).
  WHEN a unit starts no table shows, and dune hands out job slots first come, first served: a rule
  becomes ready only with its dependencies, so one that depends on a late-built input queues behind
  the whole compile. `runtest-sweep_harness` (~400 s) was main 1/2's tail that way, in every run,
  until gh-ocannl-1575: a shard-1-only step before the build compiled cc_backend's cone under a
  module log level, the build recompiled it, `tools/host_caps.exe` and the `ocannl_config` behind
  the ambient gate came ~80 s late, and the gate's dependents (the scans, sweep_harness) ran ~70%
  into the step. A step that builds under another configuration runs AFTER the shard's build (and
  its weighing, whose trace a later dune overwrites); the job log's output timestamps show a late
  cluster, since dune prints an action's output as it finishes. Every
  `main` shard job weighs its own build's trace, crediting each process to the unit its TARGET
  names -- an alias action's `runtest-<name>`, a rule's `<name>.<ext>` file -- never to its
  executable, which several tests share (`online_softmax.exe` runs three,
  `inline-test-runner.exe` a family) -- and a rule output named after no unit to the unit whose
  stanza names it, any key left unplaced reported loudly; and `tools/ci-shard.sh merge-weights` turns runs' shard
  logs, one file per run, into the replacement table, averaged over the runs (the script's header
  has the fetch recipe). Refresh it from runs of the CURRENT code when `tools/ci-times.sh` shows
  the shards drifting apart: `plan` balancing a table to the second while CI's shards ran 32 and
  14 minutes was a table measured before gh-ocannl-1340 made the autotune tests 2-3x faster. `dune show aliases`, which
  the script reads, prints on STDERR and exits 0 for a directory it does not know (and `dune
  rules` prints nothing for an action-only alias, so it cannot answer coverage questions either):
  the script validates the listing and refuses (exit 2) rather than shard a partial one.
  The `train` suite needs no dune: its units are read from the tracked dune files' `(alias (name
  train) (deps ...))` aggregates as S-expressions, and the reader refuses whatever would make its
  member list differ from what `@train` builds (a dep that is not `(alias train-<name>)` defined
  beside it, an extra aggregate field, a rule on `train` itself, an `include`).
  `test/operations/ci_matrix.sh` holds the matrix to exactly `1/N..N/N` for each sharded suite, the
  unsharded legs' alias list to the script's and the sharded step to passing its suite, and runs
  the train shards for the matrix's N against its own reading of the aggregates (they must
  partition it); `tools/test-ci-shard.sh` is the hermetic harness.
- **A red build job ends with the dune rules that failed** (gh-ocannl-1304). Every dune build step
  copies its output into `$RUNNER_TEMP/dune-build.log` through `tee`, and a failure-only last step
  runs `tools/ci-failed-rules.sh --github` over it: one entry per dune error header (`File
  "<path>", line(s) N[-M]`), the rule named by its own `alias`/`name`/`target` fields, the command
  dune ran or the error it printed, each also an `::error` annotation on the run's summary page
  whose message leads with the entry's `<path>:<lines>` (the annotation list shows the message,
  not `file=`, and a golden mismatch has no other detail to name it; gh-ocannl-1464).
  That step exits 1 deliberately, so `gh run view <id> --log-failed` includes it. The script reads
  a log GitHub serves too (timestamps, colour, CRLF, `--log-failed`'s JOB/STEP prefixes), so
  `gh run view <id> --log-failed | tools/ci-failed-rules.sh` is the first read of any red job; it
  takes log text on stdin or a log file path, never a run id. To save a job log, use
  `gh api repos/lukstafi/ocannl-staging/actions/jobs/<job-id>/logs --allow-escape-sequences > job.log`
  (the id is a job id, not a run id), then `tools/ci-failed-rules.sh job.log`. It
  names a stanza from the source tree only when every line dune quoted still matches, so a log
  from another revision falls back to dune's excerpt. A failing test's own output can hold
  dune-shaped text, so the reader takes a header only at the start of its line and drops a
  dune-file header whose excerpt omits the first or last line it names (dune quotes both); a
  harness that replays fixture logs as its failure output prefixes them (`sweep_harness.sh`'s
  `replayed`, `| `), so neither side alone carries it (gh-ocannl-1590). Out of reach: a test
  printing a bare, self-consistent dune block. A script that reports a red dune run in one line
  reads `--names` (one short name per entry; the native-PTX leg's verdict, gh-ocannl-2005) rather
  than dune's `File "..."` paths, where every failing `(test)` is its dune file. **A pipe into `tee` reports tee's status**:
  each such bash step says `shell: bash` (GitHub's `-eo pipefail`; the implicit default `bash -e`
  has no pipefail, and the build would read green), and the PowerShell one ends `exit
  $LASTEXITCODE`; `test/operations/ci_matrix.sh` holds both, with mutants.
  `tools/test-ci-failed-rules.sh` (the `promotion` harness group, so current Dune and the floor)
  builds a fixture project red and green, with dune's colour forced as setup-ocaml forces it
  job-wide (`CLICOLOR_FORCE=1` in its `build()`, whatever the ambient setting; gh-ocannl-1928): a
  leg reading a raw log without stripping the escapes fails locally as it did in CI, where two
  such reads cost lukstafi/ocannl-staging#1669 a round; a leg pins that the built logs carry them.
- **Each listed rule quotes why it failed** (gh-ocannl-1831): under its entry, behind `| `, the
  first 3 lines of its own block holding `FAIL:`, `FAILED:`, `AssertionError`, `Error:` or
  `Fatal error:` (of a golden diff, only the `+` lines), and a count of the rest; the annotation
  carries them on its one line. A block ends at the next header, at another action's `(cd ...)`
  line (CI's display prints one above any output, a passing action's too), or at a dune error
  chained to a target by `-> required by` lines (below any lines dune wrapped its message onto)
  -- dune's own, listed with its chain under
  `dune errors without a location` wherever it falls, since that chain is its only attribution
  (ahrefs/ocannl#1720 read a chained `CreateProcess()` error as the listed rule's). Dune elides
  the middle of a long action output (`...TRUNCATED BY DUNE...`), so a test prints its verdict
  last. `gh` 2.x prints a log's ESC as the text `^[` (and `gh api` refuses the raw bytes), which
  the reader strips like a raw escape: before that, a piped `gh run view --log-failed` listed
  nothing for any coloured header.
- **A listed rule with no such line shows its last output line** (gh-ocannl-1927), behind
  `last output:` so `| ` keeps its one meaning: an entry that shows the command it ran and whose
  block matches none of those patterns -- a shell error (`/bin/sh: ./x: No such file or
  directory`, the `Permission denied` rows of ahrefs/ocannl#1720), a signal, a bare exit status
  -- quotes the last
  non-blank line of its block, in the annotation too. The last, since dune keeps the end of a long
  output and the first line is usually a banner; when the action printed nothing (a test whose
  stdout goes to its `.actual`, say) that line is dune's own `Command exited with code N.` or
  `Command got signal SEGV.`. A compile error, a golden mismatch and a rule dune reported an error
  for carry their reason already and get none. In a job log GitHub serves a block also ends at a
  `##[` line (`##[error]Process completed ...` closes the build step), so a rule that printed last
  is not lent the later steps' lines -- the listing step's own echo among them. Any header ends a
  block, a dropped one too, so a test printing a bare dune block shows the line above it.
- **Each build job prints the host it drew** (gh-ocannl-1371), before building: `tools/host_caps.exe`
  (run from `test/config`) prints `Context.host_capabilities` -- CPU, CPU count and pool tag, the
  ISA the cc probes find under the kernel flags (`avx2+fma`, `avx512f+bw+vl`, `neon`), the resolved
  `cc_vector_bytes` beside the auto width, the register count, fp16 support, the parallel-Grid
  runtime, compiler and flags, the codegen tag, then any linked GPU backend's host-side probes
  (gh-ocannl-1038, where the same list became a dune input).
  GitHub's ubuntu runners come with or without AVX-512, and the auto width (64 or 32) changes what
  the autotune pre-filter keeps (gh-ocannl-1363's flake). The line is repeated above the failure
  listing. Read it first when a red reproduces on one runner and not another.
  The clean-room `ci-env.yml` builds the probe after its package build, with `--release`
  from the repository root to reuse those objects, then runs the executable from `test/config`
  (gh-ocannl-1948): `--release` implies `--root .`, so combining it with `--root ../..` is a
  usage error. Every dune step saves combined output under explicit `shell: bash` (pipefail);
  its failure listing repeats the saved host through the same `ci-failed-rules.sh --host` path.
  `ci_matrix.sh` checks the shared save path, probe executable and configuration directory,
  release profile and log capture using the shared step/command readers. It executes the
  failure listing against both reachable states: build log alone, and build log plus host.

### opam and caches in CI

- `test/operations/ci_matrix.sh` extracts and executes the three "List opam's missing
  system dependencies" run blocks from `ci.yml`, `gh-pages-api.yml` and `ci-env.yml`
  against stub `opam`/`dpkg-query` (gh-ocannl-1917). It compares the bodies apart from
  the resolver's test/doc flags, derived from each workflow's system-dependency install
  (`-d`/`-t` and their long forms), and checks installed, held, provided (including versioned
  aliases), removed and missing packages, empty sets, and command failures with partial output.
  Install/resolver flag mismatch, copy-drift and removed-provides-loop mutants exercise the oracles. The named-step
  and literal-block helpers read the workflows' current indentation and `run: |` dialect;
  another scalar style needs reader support. All three steps need explicit `shell: bash`:
  GitHub's implicit `bash -e` masks a failed `dpkg-query` behind successful `awk`, emits a
  partial cache request and bypasses the intended warning/fallback. The fixture follows the
  step's shell selection so removing `shell: bash` fails the dpkg failure case.
- **Nothing in this repository pins the opam version.** `ocaml/setup-ocaml@v3` is a moving tag,
  and the action resolves the latest STABLE opam release under its own upper bound at run time, so
  a release reaches CI as soon as that bound admits it and no commit here marks the change: opam
  2.6.0 arrived on its release day, 2026-09-17, in the first run after setup-ocaml v3.9.0 raised
  the bound from `<2.6.0` to `<2.7.0`. Which version a run actually used is in the `Installing
  opam` group of its setup-ocaml step — read it before blaming a change of ours for a CI-wide
  shift in timing or behaviour. Two consequences to expect at every bump. setup-ocaml's OWN cache
  key carries the opam version AND the repository URLs, so the first run after one rebuilds the
  root and bare switch: for 2.6.0 (which also moved setup-ocaml's default repository from the git
  opam-repository to HTTP `opam.ocaml.org`) the step went from ~70s to 278-347s once on
  ubuntu/macOS, then settled at 45-50s — below the old warm number, the HTTP repository's
  `index.tar.gz` being what opam 2.6 reads in memory instead of extracting. That is worth much
  more on Windows than the CI numbers show: measured on a native-Windows fleet box with scratch
  roots and the same HTTP repository, `opam init --bare` went 135.6s -> 12.6s and `opam update`
  102.8s -> 1.2s between 2.5.2 and 2.6.0, and the root went from 46 MB in 19,123 files to 25 MB in
  15 — an extracted tree of small files is exactly what NTFS charges for. CI's Windows cold switch
  hides it because that step is dominated by building the compiler: it stayed in its historical
  615-902s band (770s and 710s on the 2.6.0 dispatch). A cold switch is a bill paid once per bump,
  not a regression to chase. Nor does the win show up warm: the Windows setup-ocaml step went
  77s/111s under 2.5.2 to 89s/87s under 2.6.0, and the pin and resolve steps moved by seconds in
  both directions. Do not expect opam upgrades to move CI's Windows wall-clock at all — that job
  is priced by its cache restore, its dune build and, when cold, by building the compiler. Two of
  those are where to look first. The restore is the big one and it is not free even when it works:
  restoring the 893 MB Windows entry costs ~530s on every job, which is what it buys off the 6-10
  min dependency install. The transition-time CA collision (gh-ocannl-1014) is avoided by caching
  `tools/windows-opam-cache.sh`'s two tar payloads under the Windows v3 key: the switch and
  Cygwin tree retain the mingw libraries and package metadata, but exclude the derived
  `root/etc/pki/ca-trust/extracted` subtree at both save and restore. setup-ocaml owns those fresh
  CA files. Inner tar processes use `MSYS=winsymlinks:sys`: the default MSYS deepcopy mode
  cannot restore forward symlinks or Cygwin absolute targets into an empty tree. System link
  files preserve their POSIX targets without native Windows symlink privilege — but only Cygwin
  and MSYS programs follow them. **Anything a native program runs must restore as a file**: on
  the hosted runner the compiler install (under setup-ocaml's `CYGWIN=winsymlinks:native`)
  leaves `bin/flexlink.exe`, `ocamllex.exe` and friends as links to their `.opt` twins (inferred
  from a cache miss building green and every hit failing on exactly those names; reproduced on
  rog with a crafted link member), and restored as system link files they fail dune's `CreateProcess` with `This version of ... is not compatible with the version
  of Windows you're running` — every scheduled Windows job from the first v2 cache hit on
  (2026-10-06) died linking ppx drivers, masking the whole suite (gh-ocannl-1271). So the switch
  is packed with `tar -h` (links resolved, on the host where they work) and restore refuses a
  switch payload that still carries a link; the Cygwin tree keeps its links, which only Cygwin
  follows. `cygpath -w` resolves links too, so a native check must spell the link's own path
  (`cmd //c "$(cygpath -w dir)\\name"`), as the fixture's `native_run` does. A rog switch built
  without that `CYGWIN` setting holds copies, not links, so the `--real` leg cannot show this.
  actions/cache restores only the payload directory; checked listing and extraction
  then gate the job, so a partial extraction cannot masquerade as a warm switch. An ordinary
  cache miss still uses the two-step install, prunes sources, and packs before the post-save.
  `tools/test-windows-opam-cache.sh`, in the shared shell-harness entrypoint, exercises a real
  outer-cache round trip, a symlink payload over fresh regular CA files, corruption and extraction
  refusal, with pack/restore exclusion mutants. Native Git Bash runs the same fixture without
  touching the user's opam root; its `--real` leg compares a restored existing switch and Cygwin
  tree, then executes the restored compiler.
  Our own `_opam` key deliberately does NOT carry the opam version: a switch built by 2.5.2
  restores and runs green under 2.6.0 (the 2026-09-17 master runs hit that cache), so keying on it
  would buy nothing and cost a ~180-package rebuild per platform at every bump.
- **A slow ubuntu setup-ocaml step is apt, not opam** (gh-ocannl-1544). The step's spread (55s to
  11min on one day, 2026-10-22, on jobs whose opam caches all hit) is the `apt-get install` inside
  its `Initialising opam state` group: `bubblewrap musl-tools rsync` plus the optional
  `g++-multilib gcc-multilib` (setup-ocaml's `src/system-packages.ts`), 29 packages, 23.6 MB,
  nearly all of it gcc-multilib's lib32/libx32 halves, fetched from the runner's
  `azure.archive.ubuntu.com` at 39-46 kB/s on a slow runner (`Fetched 23.6 MB in 10min 2s`) and in
  1s on a fast one -- while the same runner's other apt fetches minutes apart, from the same or
  another mirror, ran at MB/s. Read the `Fetched ... in` line of that group before blaming a cache.
  It hit the Formatting and Dune-floor jobs as hard as the build shards, and once put a main shard
  at 37.5 minutes. `.github/actions/setup-ocaml-debs` now pre-installs that list from `.deb` files
  in actions/cache, keyed by apt's own `--print-uris` (file names, sizes, hashes), before every
  ubuntu setup-ocaml step in the workflows, so the action's install finds nothing to fetch; the list
  restates setup-ocaml's, so a change there costs only the speed back, and every step of it is
  best-effort. The mechanics live in `.github/actions/apt-debs-cache` (gh-ocannl-1546), which
  `gh-pages-docs.yml` uses for its 180 MB of texlive, the same slow mirror having held that job at
  its 30-minute ceiling: the key carries the runner image (`ImageOS`) and a prefix restore of the
  previous digest's files, so a changed dependency refetches only itself; the caller refreshes the
  apt index first if it needs a current one, and keeps its own plain install of the same list
  after the action as the fallback. opam's depexts take the same route before each ubuntu depext
  install (`ci.yml`, `ci-env.yml`, `gh-pages-api.yml`; gh-ocannl-1574): a step asks opam for them
  (`opam list --external --resolve=<project packages>` under the install's own test/doc flags),
  keeps those no installed package is or `Provides`, as opam's depext step does (an `apt-get
  install` of an installed package upgrades it, and of a provided name replaces its provider:
  22.04's `pkgconf` provides `pkg-config`, and the first run of lukstafi/ocannl-staging#1683
  removed `pkgconf` and `r-base-dev` by installing the real one), and hands the rest to the
  action under `opam-depexts-debs-v1`, whose plan is `--no-remove`, so a filter miss warns
  rather than removes. The fill is a step of its own with its own cap (gh-ocannl-1649):
  `apt-get --download-only` in fresh attempts of `fetch-attempt-minutes` each, keeping finished
  files, since the slowness is per apt session, not per runner -- so a 15-minute job ceiling
  never has to clear a slow fill. **`actions/cache`'s post step inside a composite action that
  another composite calls evaluates its inputs empty** (`[warning]Input required and not
  supplied: path` at job end) and saves nothing, while the same action called straight from a
  workflow saves: a cache whose restore says `Cache not found` on every run is read at its post
  step, not its restore. So the action saves through an explicit `actions/cache/save` step right
  after the fill -- which also keeps a fill that a later failure or cancellation follows (a post
  step saves only on job success) -- and only a complete set, since an exact hit is never saved
  over. The one-off `Failed to find any opam release that matches the specified version
  constraint` (2026-10-22 15:06-15:24Z, several jobs within seconds, green on rerun) is a different
  lookup: the action lists `ocaml/opam`'s releases through the GitHub API at every run and found
  no stable one in that answer. setup-ocaml v3.9.0 has no input that pins the opam version or
  skips the lookup, so there is nothing to pin here short of replacing the action; a re-run is the
  remedy, and a deterministic form of the same message is gh-ocannl-1543's.
- **Every `sudo apt-get update` in the workflows is capped and retried** (gh-ocannl-1809).
  Unbounded, the same mirror stalled the refresh itself on 2026-11-01 and held four job attempts to
  their 45-minute ceiling, a cancel that reads like a test hang (the tell: the log stops inside
  `Update apt package index`, before `actions/checkout`). Each copy caps an `apt-get update`
  session with `sudo timeout` (240s, apt's own per-connection timeout and retries inside it), gives
  a fresh session one more try on `archive.ubuntu.com` / `security.ubuntu.com` after rewriting
  Azure entries or verified `mirror+file:/etc/apt/apt-mirrors.txt` references in either
  Ubuntu source layout (gh-ocannl-2184), and ends within about 9 minutes with an `::error` saying whether it
  stalled; `ci.yml`'s matrix-job copy carries the rationale. `APT::Update::Error-Mode=any` makes
  partial index failures trigger the fallback too; a missing direct or mirror-list Azure source refuses the rewrite
  explicitly rather than retrying the same mirror. It is inline, not a local action, on
  purpose: a `ci.yml` dispatch with `expected_sha` checks out that commit, which may predate any
  new action under `.github/actions/`, so a step that `uses:` one fails the red-span bisection of
  older commits (the floor that sets: [How far back a dispatch reaches](#how-far-back-a-dispatch-reaches)).
  `workflow_job_timeouts` holds every refresh in the workflows, the composite actions and the
  `*.sh` scripts to it (gh-ocannl-1852): `timeout` written directly in front of apt, a positive
  duration; a quoted mention passes only as `echo`/`printf` text (its header states the grammar).
- **A Linux CI job that fills its disk is stopped, not left to hang** (gh-ocannl-1771). On a full
  disk dune does not fail: run 41203731123 (attempt 1) printed `dune_trace_write(): No space left on
  device` three minutes into an ubuntu main shard and then sat, `dune` and its `tee` alive, until the
  45-minute ceiling cancelled it. `ci.yml`'s step `Free disk space and start the disk watchdog
  (Linux)` deletes the preinstalled toolchains nothing uses (Android, .NET, GHC, CodeQL) in the
  background and leaves a watchdog polling free space every 5s; under a 1 GiB floor it kills every
  dune from then on, and `Report the disk headroom (Linux)` (always run) prints the end headroom and
  the lowest reading, and fails with `::error title=disk space::` on a trip. The lowest reading
  counts only from the cleanup's end, and prints next to the free space before and right after the
  cleanup (gh-ocannl-1870): the cleanup frees more than a build uses, so a low taken during it reads
  the runner's starting disk, and after-cleanup minus the low is the build's peak use (an
  underestimate by what the switch setup wrote while the cleanup ran). The tell of a trip: that
  annotation, and a build step ending in a killed dune. The watchdog's state is in `/dev/shm`, and
  its output is detached, or the step would wait on it; `test/operations/ci_matrix.sh` runs both
  steps against stub `df`/`sudo`/`pkill`; a dispatched drill that filled a shard's disk mid-build
  (run 41476763884) tripped 7s after the fill and ended the job in 4 minutes. Runner disks vary:
  that drill's runner started with 86 GB free of 145 GB, so the ENOSPC run's must have been a far
  smaller one. macOS and Windows jobs have no watchdog (no ENOSPC seen there).
- Where this repository PARSES opam's output, it freezes the format with `--cli=2.1` rather than
  tracking opam — `.github/actions/pin-revisions/resolve.sh` does it for both `opam show --raw
  --sort` and `opam pin list`. Re-verified against a real opam 2.6.0 (2026-09-17): the definition
  blocks still open with `opam-version:` at column 0 and the pin table still carries its
  `git+<url>#<ref>` column, and the whole action ran to matching digests. Do not advance that
  pin with opam: a declared CLI version is a floor on the binary (`--cli=2.6` exits 2 on an opam
  2.5.0), it buys no speed (the storage work is in the repository backend, and `opam list -A`
  times the same either way), and it would opt the parser into the layout changes it exists to
  sit out — opam 2.6.0 accepts 2.0 through 2.5, deprecating none, and 2.6 is not a CLI version. opam 2.6's one
  script-visible break, safe mode no longer forcing debug level 0, does not reach it either:
  opam's debug output goes to stderr and the script reads stdout. Upgrading a DEV machine is the
  part that is not free — opam 2.6 migrates the root from the 2.2 layout irreversibly, and an
  older opam binary cannot read a migrated root afterwards.
- Both `ci.yml` and `gh-pages-api.yml` cache the built local dependency switch `_opam`, where the
  ~180 compiled packages live; setup-ocaml separately caches opam's root and bare compiler switch.
  Their entries stay separate because the CI matrix and the fixed docs runner have different key
  shapes, but both keys include the platform, compiler, project opam files, ordinary-package
  solution digest and resolved-pin digest from `.github/actions/pin-revisions`. That action runs
  after setup-ocaml updates the repositories and after the workflows' `opam pin -n` steps. It asks
  opam's solver for the selected package/version set under the install's test/doc flags, hashes the
  normalized definitions of those exact versions too — minus the project's own packages — then
  derives every remote git pin from opam's live registry, resolves it with `git ls-remote`, and
  digests the shas. A pin on a full 40-hex commit is taken as its own revision, with no lookup:
  ls-remote matches ref NAMES only and answers a SHA with nothing, which the empty-resolution
  guard refused, so `pin-depends` could not pin a commit (gh-ocannl-1672). An abbreviated SHA
  still goes to ls-remote and fails loudly, since offline it cannot be told from a hex-named ref. The project packages are left out of the definition hash because `opam pin .`
  records the checkout's git ref in the pinned definition (`opam show --raw` prints
  `git+file://…#master` on a branch and `#HEAD` on the detached checkout every `pull_request` run
  gets), so hashing them keyed the cache by EVENT TYPE: for two days every PR run missed the
  switch master had just saved for the same tree and paid the 2–6 minute dependency build, while
  master's own runs hit — and a miss is also a fetch, which is how a rolled upstream archive
  checksum surfaced as a per-branch red (gh-ocannl-889). `hashFiles('*.opam')` already covers
  their content. When a key you expected to hit misses, diff the two runs' `Resolved opam package
  solution` and pin-spec listings first; if those are identical the raw definitions differ, and the
  `Definition digests:` listing right below the solution names the culprit — one
  `<name>.<version> <12 hex>` line per non-project definition, hashing that package's own
  `opam show --raw` block, so the line that moved between the two runs IS the package whose
  definition changed. That listing costs no extra opam call: it splits the one solver-wide
  `opam show --raw --sort` the digest already runs, on its `opam-version:` block boundary
  outside quoted strings. Canonical raw output has no comments and indents nested fields;
  its ordinary and triple-quoted strings still need quote/escape tracking, since descriptions
  can hold column-zero marker and identity lines (gh-ocannl-921). The harness pins exact
  hashes for that fixture and rejects both unquoted splitting and identity extraction.
  (~50ms for ~200 packages). A key over
  `hashFiles('*.opam')` alone is blind both to new compatible
  ordinary-package releases and to the
  branch-tracking pins
  (`ppx_minidebug#main`, `notty-community#master`, `dataprep#main`), which move while the opam files
  stay byte-identical. Deriving from the registry matters: a newly added explicit pin enters the key
  without a second caller-owned list to update. The action delegates to
  `.github/actions/pin-revisions/resolve.sh`, and `tools/test-pin-revisions.sh` exercises that exact
  production script with fake opam 2.5.2 and git outputs: sorting, duplicates, local-pin exclusion,
  project-package exclusion from the definition digest (and the loud failure when the solution
  holds nothing else), full and abbreviated commit-SHA pins, the per-definition digest listing (asserted to carry a DISTINCT hash per
  definition, so a per-run constant naming nothing would go red), color suppression, empty
  registries and failed resolutions all have fault-injected negative controls. CI runs the harness
  once, in its toolchain-free `harnesses` job, because this
  is POSIX action plumbing, not an OCaml test or a repository scan, and the fixtures need neither
  setup-ocaml nor network.
  Both workflows use exact-key restores only: cache
  lookup happens after derivation, so a prefix fallback could overwrite the live registry with an
  older switch. And both install non-Windows depexts unconditionally after restore, because those
  are system packages absent from `_opam` (gh-ocannl-809).
- Every workflow that installs dependencies sets opam's global `archive-mirrors` to
  `https://opam.ocaml.org/cache` right after setup-ocaml, before any pin or install. It was
  written against a setup-ocaml default that has since moved: through v3.8.0 the action pointed
  opam at the GIT opam-repository, whose `repo` file declares no mirror, while the HTTP repository
  at opam.ocaml.org declares one; v3.9.0 (2026-09-17) made the HTTP repository the default, so the
  mirror now arrives with the repository and the line is redundant with today's default. It stays
  as an explicit guard — one opam call, and the protection is then ours rather than a default's.
  Without a mirror a bare CI switch downloads every archive from its package's upstream
  `url` — and GitLab builds tag archives on demand with no byte-stability promise, so a pinned
  checksum can stop matching what `gitlab.inria.fr` serves until opam-repository re-pins it
  (menhir 20260209, gh-ocannl-889: `Bad checksum`, exit 40, on every leg that had to fetch, while
  legs with a warm `_opam` cache sailed past it). opam consults archive mirrors BEFORE the
  upstream URLs, keyed by checksum (`<mirror>/md5/<xx>/<hash>`), and the cache holds every archive
  opam-repository's CI has verified, so a package still falls through to upstream only when it is
  absent there. Reproduced with a scratch `OPAMROOT`: a package whose `url` points at an invalid
  host but carries a cached archive's md5 installs with the mirror set and exits 40 without it. It
  is the set form, `opam option 'archive-mirrors="…"'` (a quoted opam string — the bare URL is a
  parse error), because `+=` on a restored opam root would accumulate a duplicate per run. Not an
  answer to a package genuinely missing from the cache, and not a reason to drop menhir: the
  einsum parser (`tensor/parser.mly`) is what `ppx_ocannl` and the tensor library are built on,
  and the exposure is generic to every GitLab-hosted package, not menhir's.
- `ci.yml` sets `DUNE_CACHE: disabled`, on purpose. Dune's default mode,
  `enabled-except-user-rules`, excludes OCaml compilation from the shared cache by construction
  (`module_compilation.ml`'s `cm_kind_can_go_in_shared_cache` is true only for Melange, and under
  that mode every other rule defaults to not cacheable), so the dune cache setup-ocaml's
  now-deprecated `dune-cache: true` input saved for this job was a ~400-byte empty tar on every
  platform, and the "compilation still caches" premise the setting was kept under was false —
  check the post step's `Cache Size` line before believing any CI cache works. The other mode,
  `enabled`, caches user rules, i.e. the test executions whose output depends on the runner's C
  toolchain, ISA and core count — a recorded result standing in for an execution is a false green.
  A compile-only cache is possible but is its own design: dune's rule digest includes the
  cacheability flag, so a `@check` walk against a shared cache root followed by a test walk with
  the cache disabled rebuilds everything; the second walk has to stay `enabled` against a
  throwaway `DUNE_CACHE_ROOT` instead, which also loses the single-walk overlap of compilation
  with the serialized training chain. That design is gh-ocannl-901.
- **The cc census listing cache is one snapshot for every job, saved only when useful**
  (gh-ocannl-1575). A census entry is valid whoever restores it (`ci.yml`'s comment above the step
  says why), so each ubuntu `main` job restores the newest `cc-march-census-v1-<os>-` entry,
  whichever shard saved it -- keyed per shard, a piece the deal moved recompiled cold (241 s). A
  job saves (a new immutable entry of the whole directory) only when it published a listing file
  the restore did not bring, or, off pull requests, when the snapshot it restored is from an
  earlier UTC day: a hit refreshes its entry's mtime, which the census's 30-day expiry reads, and
  a snapshot nothing re-saves keeps the old ones. Saving every run had left 159 entries, 5.3 of the
  repository's 10.7 GB of caches, most of them pull-request-scoped (staging#1464 first saw the
  repository over its 10 GB). The job log's `cc census listings:` line says which way it went.
- `gh-pages-api.yml` runs for pull requests and `workflow_dispatch` as well as pushes to master, but
  only a push deploys; validation runs receive their own concurrency group and exercise the build
  and cache topology without publishing. Both Pages workflows split a push run into `build` and
  `publish` (the site passes as an artifact). Only `publish` joins the shared `gh-pages-deploy`
  group with `queue: max`, so the two never race over the gh-pages branch and neither evicts the
  other's pending publication (gh-ocannl-808, gh-ocannl-825). Each workflow's run sits in a group
  of its own (`gh-pages-docs-build`, `gh-pages-api-build`) with GitHub's default queue -- one
  running, the newest pending -- because a newer push of the SAME workflow rebuilds a superset,
  while a newer push of the other one would not (the path filters differ). The whole run
  in the shared queue, as before, put one docs build's 30-minute texlive fetch ahead of twelve
  later push runs, the base gate's `github pages api` verdict at the tip among them (gh-ocannl-1546).

### How far back a dispatch reaches

- A `ci.yml` dispatch with `expected_sha` runs the workflow as the dispatched ref spells it, over a
  checkout of the named commit (gh-ocannl-1853). A step that `uses:` a local action
  (`./.github/actions/<name>`) loads it from that checkout, so a commit older than the action, or
  than the interface the workflow calls it with (the inputs it passes, the outputs it reads), fails
  that step, and each such step runs before its job's build and tests. The newest such action is
  the bisection FLOOR: below it, a dispatched job that uses one fails before it builds, which bounds
  the red-span bisection under [What CI actually covers](#what-ci-actually-covers). A job-level
  `uses:` of a reusable workflow is loaded from the dispatched ref, not the checkout, and bounds
  nothing.
- `Verify dispatch commit` fetches `ci.yml` at `GITHUB_WORKFLOW_SHA`, derives its local
  `uses: ./...` paths (optionally quoted), and refuses a checkout with neither `action.yml`
  nor `action.yaml` for any of them, naming the branch-dispatch route below (gh-ocannl-1936).
  The derivation covers every job's actions, but only the `run` job reports the refusal before
  its setup. On an ordinary dispatch, parallel sibling jobs can still fail at their own local
  action steps with GitHub's bare error; the `run` job's annotation explains the floor. The
  recorded 5.3.x drill skipped those siblings. An empty derivation is refused too. This tests manifest
  presence, not interface compatibility or the later `run:` steps' script/flag bounds.
  `ci_matrix.sh` takes the expected actions from the floor record below (held equal to the
  workflows by `workflow_job_timeouts`), then executes the guard against each absent action,
  quoted paths and new actions in another job;
  a mutant dropping one action from the derivation must fail that same check.
- The record below lists each local action the dispatched workflows' steps use, oldest first, as
  `` - `<action path>`: YYYY-MM-DD, lukstafi/ocannl-staging#N. `` -- the master merge that first
  carried the action with the interface the workflows use, and that merge's date -- and then
  `- Floor: YYYY-MM-DD, lukstafi/ocannl-staging#N.`, which is the newest entry's mark.
  `workflow_job_timeouts` holds it to the workflows: the local actions the steps of every workflow
  naming `expected_sha` use must be exactly the listed ones, each listed once, the dates must not
  decrease, and the floor must equal the last entry. A new local `uses:` there therefore fails
  until it is listed last and the floor moves to it; that is the cost which kept gh-ocannl-1809's
  apt step inline (above). The scan reads no git, so it cannot tell a wrong date or merge, and it
  does not compare interfaces: a change to a listed action that a workflow's new call depends on
  moves that action's entry, and the floor, by hand.
- The floor's merge is master's first-parent commit titled `Merge pull request #N from ...`, and a
  candidate commit is at or above it when `git merge-base --is-ancestor <that merge> <candidate>`
  succeeds. The floor bounds every job's setup, not its later steps: a `run:` step names tree
  scripts, and the flags it passes them, as the dispatched ref spells them, while the scripts are
  the checkout's. A dispatched job red on a missing file or an unknown flag is therefore no verdict
  on the commit, and nothing holds those per-step bounds.
- Below the floor, or below a step's bound, dispatch the commit's OWN workflow: push it to a
  branch of its own (`git push origin <sha>:refs/heads/bisect/<short-sha>`), dispatch `ci.yml` on
  that branch without `expected_sha`, so the checkout is the branch head and the workflow and the
  tree agree, and delete the branch afterwards. That reaches back to the first `ci.yml` declaring
  `workflow_dispatch` (`c83d67fbb`, 2026-08-12).
- `./.github/actions/pin-revisions`: 2026-08-29, lukstafi/ocannl-staging#535.
- `./.github/actions/setup-ocaml-debs`: 2026-10-23, lukstafi/ocannl-staging#1353.
- `./.github/actions/apt-debs-cache`: 2026-10-23, lukstafi/ocannl-staging#1359.
- Floor: 2026-10-23, lukstafi/ocannl-staging#1359.

### Windows CI

- Windows CI runs independently on the daily schedule, together with an ubuntu job
  on the OCaml floor the opam files claim (`>= 5.3.0`, against 5.5 everywhere else).
  PR, push and ordinary `workflow_dispatch` runs use the Linux/macOS matrix.
  When a change needs Windows signal, AGENTS.md's *Windows verification placement* says where it
  comes from: rog rebooted into Windows by the agent itself first, then a box the user boots, this
  dispatch last. The order is latency: `rog-nv-win` and `minix-amd-win` normally boot Ubuntu, and
  one rebooted into Windows answers in minutes where hosted Windows CI takes one to three hours;
  the unattended reboot also spends none of the user's attention (gh-ocannl-1042). A wave
  coordinator batches the reboots and the boot requests and picks the box with less queued work.
  The unattended reboot is `wake-lab.sh boot-windows --as=<request_id> rog` (ludics-lite#353):
  a one-boot UEFI `BootNext` into Windows Boot Manager — BootOrder stays Ubuntu-first and GRUB is
  never written — then a wait of up to 15 minutes until native Git Bash answers. Its live witness
  on rog (2026-09-24) reached Git Bash about 50 s after the reboot, with nobody at the keyboard
  and no BitLocker prompt. `<request_id>` names your own active `measurement` reservation on the
  box (`fleet-worker.sh execution reserve`, `.request.kind == "measurement"`,
  `execution_host: rog-nv-linux`): the script refuses without one, refuses while any OTHER
  reservation names the box, while its lab lane or hold lock is held, or while a logind sleep-block
  inhibitor shows a run in progress, since a reboot kills everything on the box — including the
  desktop session's open apps. Verdicts: exit 0, Git Bash answers — run the check as below;
  exit 1, refused, or the box is back in a known OS (the firmware ignored `BootNext`, or Windows
  restarted) — fall through to asking the user; exit 3, `NEEDS A PERSON`, nothing answers or the
  old OS never went down within the deadline — tell the user at once (only someone at the box can
  tell a Windows update from a BitLocker prompt or a hang; the script never retries) and fall
  through to the next step. Afterwards ALWAYS run `wake-lab.sh boot-linux --as=<request_id> rog`
  (a no-op when the box is already in Ubuntu; from Windows it is a plain `shutdown /r` into
  BootOrder's first entry), then conclude the reservation: rog's steady state is native Ubuntu,
  which the daily sweep's CUDA leg and every Linux worker there expect. Only rog qualifies today,
  and not for want of setup on minix: minix has the `/etc/sudoers.d/50-fleet-boot` grant (the two
  `efibootmgr` forms and `systemctl reboot`, as on rog; text in ludics-lite's README), but on
  2026-10-14 `boot-windows` there came back in Ubuntu with exit 1 ("came back in Ubuntu") because
  its firmware ignores `BootNext`, and its Windows, booted by hand, stopped at the BitLocker
  recovery-key prompt, so even a firmware that honoured the one-boot entry would leave the reboot
  waiting on a person. tuf has no wired NIC, so the script refuses it. So a minix Windows leg is a
  user-booted one, and on that boot `minix-amd-win`'s PATH names a `hip_path_link` folder that no
  longer exists: with hipjit installed every test executable, cc-only ones included, then fails
  to start with `0xc00004bc` (`hiprtc0701.dll` not found) unless the leg script prepends
  `C:/Program Files/AMD/ROCm/*/bin` to PATH (ahrefs/ocannl#1349; staging#1120 did).
  The fallback waits until the request is withdrawn -- the user told, through the same channel, that CI is taking the check, so a late
  reboot is not wasted -- because a dispatched run's checks gate the head, and cancelling one when
  the box turns up leaves a no-verdict the merge refuses. On a booted
  Windows host, use native Git Bash in an isolated checkout of the intended commit, source
  `tools/opam-env.sh`, and run the relevant aliases through `tools/test-run.sh`, keeping host,
  SHA, command and exit sentinel as evidence (driving rog over ssh: the leg recipe below); WSL
  does not establish native Windows coverage.
  The fallback dispatches `ci.yml` on the chosen branch with
  `windows_only: true` and the full `expected_sha`; this opt-in runs only Windows main/training,
  without duplicating Linux/macOS, formatting or floor jobs (the floor job has its own
  dispatch, `ocaml_compiler: 5.3.x`; the OCaml and dune traps section says when). Each job rejects a missing or
  mismatched intended SHA before dependency setup. Verify the run's `head_sha` and both jobs'
  executed results before counting that evidence; an existing commit needs no new push.
  Do not wait for the independent scheduled sweep to merge a PR. Ordinary current-head PR checks
  remain required.
  `test/operations/ci_matrix.sh` evaluates the actual matrix expressions,
  pins the opt-in default and trigger separation, evaluates every `ocaml_compiler` option (the
  default is the ordinary matrix, any other one ubuntu job, and every compiler only the schedule
  builds must be dispatchable as exactly its scheduled job), and exercises the dispatch commit guard. It
  reads the side jobs from `ci.yml` (every job but `run` and `notify-triage-routine`), so a new
  job lacking the fallback guard fails it with no list to extend.
  Daily, so a scheduled red spans about a day of merges rather than the 50-150 a twice-weekly
  cadence left, and the Windows opam cache stays well inside actions/cache's 7-day eviction.
  Windows and the OCaml floor ride the same cadence because they fail slowly through the
  dependency cone or toolchain.
- The Windows job ends with a smoke of `tools/test-run.sh` itself, from Git Bash: `run`, `status
  last`, a deliberately failing target, an unknown alias, `start`/`wait last`, `list`, each
  asserted against its documented exit code. That script's MSYS branches (unconditional and fatal `opam-env.sh` sourcing,
  dune routed through `dune-quiet.sh`, the tokenless liveness degradation where MSYS `ps` has no
  lstart/state columns, the flock through MSYS perl) execute nowhere else, and are otherwise reached
  only from rare hand-run Windows sessions — where their rot is discovered mid-task. It builds
  `tools/promote.sh`, a source file dune copies rather than compiles, so it costs a workspace scan;
  the failing-target leg is the load-bearing one, since it is where a dropped `PIPESTATUS` in
  `dune-quiet.sh` would report red runs as green. Its target is a file dune cannot build, never an
  alias: `test-run.sh` refuses an unknown alias itself (exit 2) before dune runs, so an alias there
  never reaches `dune-quiet.sh` (gh-ocannl-1589). It runs even when `dune runtest` above went red
  (Windows runs once a day; a golden drift must not mask the runner's health for a day) and
  it runs last, so a broken runner cannot abort the sweep's only Windows test coverage.
- That smoke step has itself been run on real MSYS (rog, Git Bash, under GitHub's exact
  `bash --noprofile --norc -eo pipefail`), so it is a confirmed test rather than an untested one.
  All four MSYS branches were checked directly and hold: `opam-env.sh` sourced from a PATH with
  neither `dune` nor `ocamlfind` yields a toolchain that LINKS (not merely compiles), and refuses
  fatally with exit 2 when opam is absent; `dune-quiet.sh` preserves dune's status through its
  stderr filter (1, 0, and an arbitrary 7 from a shim) while dropping only the `.drectve` lines;
  MSYS perl's flock genuinely excludes a second process; and `proc_alive` does not in fact need
  its tokenless fallback there — MSYS `ps` has no `-o` at all, but MSYS *does* provide
  `/proc/<pid>/stat`, so `ps_token` takes the Linux branch and records real tokens.
- A native Windows launcher using `bash --noprofile --norc` must prepend
  `PATH=/usr/bin:/mingw64/bin:$PATH` before sourcing `tools/opam-env.sh`: a PowerShell-inherited
  PATH can omit even `uname`, preventing the MSYS detection from running (gh-ocannl-770 fleet
  verification). Invoke `C:/Program Files/Git/usr/bin/bash.exe` explicitly where `bash` resolves
  to the WindowsApps WSL launcher.
- **A leg on `rog-nv-win`, driven over ssh** — the recipe the 2026-10-13 wave settled on
  (staging#1086, staging#1088):
  - **Write the leg as a script and feed it to Git Bash on stdin**:
    `ssh rog-nv-win '"C:/Program Files/Git/usr/bin/bash.exe" --noprofile --norc -eo pipefail -s' < leg.sh`,
    the script opening with the PATH prepend above. `-eo pipefail` is what CI's Git Bash steps
    run under, and without it a failed step (the cache fixture, say) is masked by a later green
    one in the leg's exit status. A leg that must go on past a red step records it
    (`step || rc=1`) and ends with `exit "${rc:-0}"`. The evidence is still each
    `tools/test-run.sh` run's own verdict and `exit:` sentinel, never the ssh exit alone. Do not
    quote the command through cmd.exe (`ssh rog-nv-win 'bash.exe -c "..."'`): that adds a quoting layer that does not treat a
    backslash as an escape, so staging#1086's escaped `\$PATH` reached bash literally, the leg's
    PATH lost its inherited entries and `tools/opam-env.sh` found no `opam`. On stdin, bash is the
    only parser, and the script reads exactly as it would run locally.
  - **The checkout**: rog's Windows-side clone is `C:/Users/lukst/GitHub/ocannl-staging` — there
    is no `~/ocannl-staging` on that side, unlike its Ubuntu boot. Fetch there and make a detached
    worktree of the intended commit under `C:/Users/lukst/ocannl-staging-worktrees/`. Run
    directories land under `C:/Users/lukst/.ocannl-test-runs/`. These paths are machine-local, but
    every Windows leg on rog needs them.
  - **Pin `core.autocrlf=false` for every git call in the leg**, starting with
    `git -c core.autocrlf=false worktree add --detach ...` (or export
    `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.autocrlf GIT_CONFIG_VALUE_0=false` at the top of the
    script, which also reaches the git calls tests make themselves). rog's system git config sets
    `true`, which checks sources out CRLF. Hosted CI sets it `false` before checkout (`ci.yml`'s
    `Set git to use LF`), and under `true` `refusal_control_scan_cases` fails, since it edits
    `verdict.ml` by `\n`-anchored patterns (ahrefs/ocannl#1311 tracks making such tests
    line-ending-proof). A leg on a CRLF tree is not the check CI runs.
  - **Run `tools/test-windows-opam-cache.sh` before sourcing `tools/opam-env.sh`**, or confine
    the sourcing and the build to a subshell of their own; a subshell started AFTER the leg's
    shell sourced it inherits the modified PATH and does not help. The switch's `bin/` holds
    opam's Cygwin `cygpath.exe`, and once `opam-env.sh` puts that directory on PATH it shadows Git
    Bash's, answering with `/cygdrive/...` paths Git Bash cannot open. staging#1088's first leg ran
    the fixture after sourcing and had to discard its result. CI's cache steps never source
    `opam-env.sh`, so this order is also CI's.
- **A Windows opam-cache key bump is verified only by the SECOND hosted run after it.** The first
  run misses the new key, so it builds the dependencies and exercises `windows-opam-cache.sh pack`
  and the save; only the next run restores the entry, and the restore is where the v2 payload
  broke (the links above). A change to what is packed (with any restore that depends on it)
  must therefore bump the key itself: `ci.yml` runs the pack step only on a miss, so under the
  old key both runs hit the existing entry and neither exercises the new pack path. The bumping
  change then
  dispatches `windows_only` twice at its head, the second only after the first has finished (the
  entry is saved in a job's post step), and counts the second run's cache hit, read in its
  `Restore opam dependency archives (Windows)` step. staging#1088 needed runs 38399107756 (miss,
  pack, save) and 38410117323 (the first v3 hit). A green first run proves the cold path only.
- The Git Bash requirement is specifically MSYS, not any bash on the box: opam's cygwin bash — or
  whatever `bash` resolves to once opam's cygwin is on PATH — reports `OSTYPE=cygwin` exactly like
  Git Bash, so the two are told apart by `uname -o` (`Msys` vs `Cygwin`). Only the MSYS one gets
  `tools/opam-env.sh`'s path rewrite (the un-primed signature is dune found but linking dead with
  `cygpath: error converting … -lpthread`), and Cygwin ships no `perl`, which `tools/test-run.sh`
  needs for its lock, cap and `last` pointer and refuses without (gh-ocannl-662).
- What that run FOUND: under MSYS with the default `winsymlinks` mode, `ln -s` does not create a
  link, it silently COPIES — so `ln -sfn <run-dir> last-<key>` left a full copy of the run
  directory where the pointer belonged, and every `last` lookup afterwards resolved to nothing.
  `test-run.sh` now points `last` with a plain file holding the path (written through a temporary
  and renamed, so still atomic), which needs no symlink privilege on any platform. Reach for a
  pointer file, not a symlink, in anything that must work from Git Bash.

## The cross-machine sweep

- Sweep endpoint and lock-contract helpers live in `tools/lab-map.sh`, registry reads in
  `tools/fleet-registry.sh`, diagnostic normalization in `tools/sweep-fingerprint.sh`, the
  comparison cursor in `tools/sweep-unit-state.sh` (gh-ocannl-990), the fleet's GPU tokens
  (a GPU unit's readings and its take) in `tools/gpu-tokens.sh` (gh-ocannl-1177), and the process
  supervisor every unit leg runs under -- `capped`, `remote_capped`, `capped_perl` and the `sq`
  quoting its far-side text ships with -- in `tools/sweep-supervisor.sh` (gh-ocannl-1380), so a
  fixture or a live witness on a GPU box sources the shipped text instead of restating it or
  cutting it out of `tools/sweep.sh` with sed. Their caller-owned
  variables and callback dependencies are listed at each file's head. `runtest-sweep_components`
  exercises the sourced functions without running a sweep; `bash test/operations/sweep_components.sh
  state` (also `lab`, `fleet`, `fingerprint`, `tokens`) isolates one fixture. The default runs all four plus broken
  component controls; `runtest-sweep_harness` remains the integration check.
  Its `sweep_remote_fixture.sh` local SSH mode executes the emitted commands with isolated
  homes and cloned repos: prep and real locks, boot-specific PATH and widths, collected kernel
  evidence, guest-identity retries and serial reruns. The harness header audits unreachable
  decisions; `sweep_harness.sh --remote-only` isolates these command-path fixtures.

- `tools/sweep.sh` is the coverage for every backend CI does not run: cc and metal locally
  (the macOS host), cuda on rog, hip and then multidev_cc on minix, and hip again on tuf, all pinned
  to ONE resolved commit so a mid-sweep merge cannot leave the machines testing different trees.
  The two GPU boxes are dual-boot, so a remote unit's ssh destination is resolved per run from the
  site's wake-lab host table (`kind_of rog` → `linux` means `rog-nv-linux`, `wsl` means
  `rog-nv-wsl`; `WAKE_LAB_HOSTS` moves the table, `OCANNL_TOOL_SWEEP_DEST_ROG`/`_MINIX` override
  it for one run), and a box whose kind cannot be read refuses the run at startup rather than
  guessing: the hard-coded `-wsl` aliases filed every GPU unit as `skip (unreachable)` — a sleeping
  box's row — from the first native boot on (gh-ocannl-1030). Only a selected remote unit is
  resolved, so a local-only run needs no table; the header's `destinations:` line names the boot
  each GPU lane addressed.
- A unit whose backend field is not a backend is a configuration VARIANT (gh-ocannl-1778), keyed by
  its own name in logs, history, unit state, `--only` and the run record, and mapped back to its
  backend wherever the backend decides (`unit_backend`, `unit_dune_env`, `unit_suite`). Three so
  far, all after cc and metal in the local lane. `cc-outlined` runs `@runtest` on cc with
  `OCANNL_CC_KERNEL_PART_WEIGHT=1 OCANNL_CC_COMPILE_UNIT_MIN_BYTES=0`: every kernel large enough
  outlined, so a kernel-text reader added after the per-test `-outlined` reruns is still checked
  against outlined text. A VALUE golden that moves there is a finding (outlining runs the same
  statements in order); a golden holding kernel text moves by design. The two keys are in
  the ambient record, so the local cc unit and this one each rebuild the other's test runs.
  `cc-gc-backprop` (gh-ocannl-2062) runs all of `@runtest` on cc with
  `OCANNL_DEBUG_GC_BACKPROP_ROOTS=true`, exposing accidental dependence on a dropped consumer
  remaining uncollected at the backprop hand-back gate.
  `cc-oracle-widen` (gh-ocannl-1964) builds one alias on cc,
  `@test/operations/slow-virtual_inliner_oracle-widen`: the inliner oracle's full cross of every
  write with every read on every shape (about 9.3 million programs with the nested-source family), which
  the default run `dune runtest` and CI hold thins on the large shapes. That cross is the only run
  holding every pair there to the semantics, and the oracle's first run found a silent wrong value,
  so it runs daily; the alias is a `slow-` member, so `tools/test-run.sh affected` names it to a
  change reaching the inliner, and on Sunday the cc unit's `@slow` builds it first (the variant then
  finds it built: same worktree, same environment). A variant takes no `@train` or `@slow` and is
  left out of the skip aggregation, which covers backends. The sweep harness pins each variant's
  one dune call (`sweep_harness.sh`, the `gh-ocannl-1778`, `gh-ocannl-2062` and `gh-ocannl-1964`
  cases) and the run record's backend map.
  Each machine's units form a lane — sequential within the box, which shares one worktree, lock and
  device, concurrent with the other boxes' lanes (gh-ocannl-976) — so wall-clock is the longest lane
  (the local one, metal's suite), and the remote units start while the routine's freshly woken WSL
  VMs are still up. The CPU pair is split across boxes on purpose: it takes a unit off that longest
  lane, and runs a CPU backend's goldens under Linux as well as macOS every day. Rows reach the
  history in completion order, each appended under a lock. multidev_cc needs no hardware
  and is there anyway: it keeps its own `micrograd_demo_logging` debug-log golden, which
  `dune runtest` diffs only under `OCANNL_BACKEND=multidev_cc` — a spelling neither test/config's
  pinned `backend=cc` nor CI ever sets. gh-ocannl-700 is the cost of that gap: `eefa827e`
  (gh-ocannl-461) reordered backprop fragments, re-promoted the cc golden, and left multidev's
  stale and master red under that backend for six weeks. **A backend with its own goldens and no
  leg here is a silent regression channel, hardware or not** — add the leg when you add the
  goldens. That multidev_cc's leg lives here rather than in CI is a decision, recorded at the
  runtest step in `ci.yml` (gh-ocannl-756): a second per-PR runtest leg would roughly double the
  ubuntu job's ~28min test time, while the daily sweep gates the backend at a ~30-commit bisect
  window without taxing any merge. The sweep records a row per unit in
  `~/.ocannl-sweep/history.tsv` and never
  exits non-zero for test failures — its exit code is not a verdict, the history file is. A daily
  scheduled task drives it.
- **hip runs twice, once on each side of the unified/discrete line** (gh-ocannl-1035): minix's
  gfx1151 is an iGPU whose device memory IS host memory, so a missing or misplaced host↔device
  transfer can read the right bytes there, and only tuf-amd-linux's discrete RX 7700S (gfx1102)
  exposes it (ludics-lite#320). `--only hip` selects both units; each is its own row, keyed by
  machine, and `known_backends` lists hip once. tuf is a Wi-Fi laptop the caller cannot
  wake (its own `WakeSystem=true` timer does, from self-improve's Linux bootstrap, or a person), so
  its lane is **gated** (`lab_box_gated`): it asks `wake-lab.sh status tuf` (`OCANNL_TOOL_SWEEP_WAKE_LAB`
  moves the script) before reserving or dialling anything, and a box not at `linux=UP` — or one
  that answered status and then not the unit — records outcome `gate`, not `skip`: nothing tested,
  and nothing the caller was expected to repair. A lane that reached tuf ends with
  `wake-lab.sh sleep tuf`, AFTER closing its own lane lock (wake-lab's power verbs take the box's
  lane lock and would be refused by the lane asking); a refusal (an inhibitor, another session's
  lock) is reported as `left awake, not a failure`. A lane that reached tuf and then stopped early
  (cancelled, or its own `die`) asks from its EXIT trap instead, detached (`setsid`, so the group
  TERM that cancelled the run does not take it) into `logs/<stamp>-tuf-sleep.log`; the relay stops
  taking signals once the unit is reaped, since the top level's own relayed TERM would otherwise
  end the lane inside that trap. The lane's line saying so reaches the sweep's output: a lane's EXIT
  trap prints nothing (it runs with the stdout of the leg it interrupted, the unit's log, where such
  lines used to land), and the top level, which waits for every lane before it removes their
  coordination directory, prints what each one buffered (gh-ocannl-1039). Its unit is uncapped:
  lukstafi/ludics-lite#344's ladder ran it green at dune's default (16) and at `-j 8`, 1046 s
  against 1105 s, with no GPU kernel line (it had run at a placeholder `-j 2` until then). Its boot kind comes from
  the site table's `kind_of tuf` like every box's (wake-lab.sh requires one too): the lab's map lists
  a `tuf-amd-wsl` the lane never dials.
- **A native lane's work legs hold a logind sleep inhibitor** (gh-ocannl-1035). Every far-side leg
  that takes the worktree lock — preparation, the suite, the RTC context, the serial rerun — runs
  under the perl supervisor's `--hold <why>` on a `-linux` destination (`sleep_guard_why`), so
  another session's `wake-lab.sh sleep`, or a laptop's idle policy, is refused by the OS while a
  unit runs; `systemd-inhibit --list` names it `ocannl-sweep` with `ocannl sweep <stamp>
  <machine>/<backend> <leg>`. A `-wsl` leg takes none (a guest's inhibitor cannot hold its Windows
  host; the Windows-side holder does). The guard is fleet-worker.sh's `execution hold` design — an
  inhibitor HELPER beside the unit on a lifetime pipe the unit's tree inherits, never
  `systemd-inhibit -- <unit>`, which closes fd 9 (the worktree lock), rewrites a signal death to
  exit 1, and is what an ssh drop hangs up on — restated in the supervisor rather than called:
  tuf's skills checkout was at a revision with no `execution hold` on 2026-09-23 (a box outside the
  worker roster is never refreshed by the fleet's preflight), where calling it would have filed
  every unit as a red suite. Fail-open and loud: a missing inhibitor or polkit grant writes
  `sweep-hold: WARNING` into the log and the unit's summary line says `ran WITHOUT a sleep guard`.
  The harness runs the extracted supervisor against fake inhibitors (held for exactly the unit's
  lifetime, refused, absent, not asked). Residual: the seconds BETWEEN legs are unguarded.
- **A lane coordinates with the fleet through two things it does not own**: wake-lab's lab locks
  and the fleet's execution registry (both lukstafi/ludics-lite). A remote lane holds the box's
  LANE lock (`$WAKE_LAB_LOCK_DIR/<box>.lock`, never `<box>.hold.lock`; the comment above
  `LAB_LOCK_DIR` in `tools/sweep.sh` says why each) for its whole length, so wake-lab's destroyers
  (`restart-wsl`, `sleep`, the boot verbs) refuse the box under it. **The sweep keeps no table of
  ssh aliases** (gh-ocannl-1121): a run with a selected remote unit reads `wake-lab.sh endpoint-map`
  once at startup, and each unit's destination is the `-linux`/`-wsl` alias (wake-lab's own suffix
  rule) on its box's row for today's `kind_of`, the box an alias reserves is the row that lists it,
  and a lane's registry names are that whole row. No wake-lab.sh, no map, no row for the box, or no
  alias for today's boot refuses the run (startup exit 2, no record); a local-only run asks
  nothing. A renamed alias needs no sweep edit. The old hand-written `lab_dest_of`/`lab_box_of`
  rows, their agreement check and the no-map fallback's stem rule cost three review rounds on one
  class of finding (staging#868) — do not reintroduce a copy. What stays checked at startup against
  the wake-lab.sh the run will meet (gh-ocannl-1025) is `wake-lab.sh lock-path <box>` answering the
  file the lane opens; a disagreement refuses the run naming what moved. ludics-lite's
  `scripts/test-wake-lab.sh` checks the same contract from its side, against a staging checkout it
  never fetches. Separately, before EACH unit the lane reads the registry
  (`fleet-worker.sh execution list --active --compact`, through the fleet-worker.sh
  `tools/fleet-worker-candidates.sh` finds, the one `tools/test-run.sh` takes its slot through) and
  records `skip (box <box> under an exclusive measurement: <request_id> (<state> on <host>))`
  when an outstanding `measurement` names any endpoint of the box — the fleet name `execution slot
  --probe` gives for the local lane (gh-ocannl-1097: the rog lane's cuda `@slow` ran into a 4-7 h
  tuned measurement and contaminated every timing after it). Correctness reservations defer
  nothing (the fleet lets them share a box, and a standing one lasts a worker's life). An
  unreadable registry fails open with a `WARNING` on the unit; no fleet-worker.sh, or
  `OCANNL_TOOL_FLEET_WORKER=none`, prints `reservations: NOT CONSULTED`. It is a read before each
  unit and the sweep owns no registry record; a measurement reserved while a unit RUNS is refused
  fleet-side instead (lukstafi/ludics-lite#451, closing lukstafi/ludics-lite#445): a `measurement`
  reserve, run or dispatch refuses a box whose wake-lab LANE lock is held, naming the holder, and
  holds that lock SHARED while it writes its record, so a remote lane's take-lock-then-read order
  closes the race both ways. The local lane takes no lab lock, so on this host nothing refuses a
  measurement reserved mid-unit; the per-unit read is its only check. The harness pins its
  registry reader to a fake, since the default candidates reach the real registry on a fleet box.
- **A remote GPU unit records its box's GPU-token holders, and where the tokens bind it takes one**
  (gh-ocannl-1177; `tools/gpu-tokens.sh`). The fleet bounds a box's GPU with GPU tokens, the first
  T of its S correctness-slot files (lukstafi/ludics-lite#391; rog-nv-linux 4 and 2, because three
  concurrent cuda batches ran out of device memory). The sweep took none, so a sweep unit could be
  the third batch. Now every remote GPU unit asks the box's own `fleet-worker.sh execution slot
  --probe` (the same candidate list as this host's, under the box's home) and, from the same
  script, which slot files are held and by whom: `execution slot --holders` (lukstafi/ludics-lite#639;
  gh-ocannl-1417), so the fleet owns the one read of the kernel's flock table and the sweep parses
  none. At its start (`opened`), after its suite (`closed`), and when a take is refused (`refused`),
  one JSON line each in `<log stem>.gpu-tokens` (`box`, `event`, `at`; per holder the fleet's own
  line -- `slot`, `kind`, `pid`, `user`, `command`, `cwd`, `source`, `unknown`, the vocabulary of
  the fleet's co-tenant records, lukstafi/ludics-lite#566 -- plus the sweep's `gpu`), beside a
  `reservations` line with the registry's active reservations on the box (from the per-unit
  `execution list --active` read; a reservation holds no token, so the counts come from the flocks).
  A reading that saw another batch holding the GPU is named on the summary, the record's twelfth
  field counts them (`opened=1/2,closed=0/2`), and the fingerprint says only whether there was
  contention. Where T < S -- rog-nv's cuda unit today -- the suite, the serial rerun and the
  completion pass each run inside one token, through `execution slot --gpu --wait
  ${OCANNL_TOOL_SWEEP_TOKEN_WAIT:-1800}` on the box; a token that does not come free in time is a
  `skip (no GPU token on <box>: ...)` naming the holders (never a third batch), a measurement
  refusing the take is a `skip` too, and a take the fleet's machinery cannot answer runs the leg
  without a token under a `gpu token: WARNING` line, failing open as the registry read does. The
  RTC context, the box check and the diagnostic legs not marked `gpu` hold no GPU test and take
  none (the `gpu` ones take theirs, gh-ocannl-1705); workers' batches and the GPU
  integration loop (lukstafi/ludics-lite#615) wait for the same tokens, with no precedence either
  way. `OCANNL_TOOL_FLEET_WORKER=none` reads and takes nothing. The fleet's `kind` is the SLOT's:
  on a box whose tokens do not bind every slot is `gpu`, so a `--cpu` batch there, or one that fell
  back into a token slot, is listed `gpu`; the sweep counts a holder as on the GPU (`gpu: true`) only
  when its command does not declare `--cpu` before the first `--`, as the reader it replaced did --
  a declared CPU batch holds no device memory, and a launch OOM is what the count attributes. A
  `--holders` that exits non-zero (a skills checkout from before ludics-lite#639, as tuf's can be)
  or prints anything but holder lines for the probed box is an `unread` reading, never zero holders.
  Only remote GPU units read, so the local metal unit records `-`; a token taken and released
  between two readings is not seen.
- **Each lane runs `benchmarks/check-box.sh --box-only` once** (gh-ocannl-1062), after its first
  unit whose suite ran (`pass`, `incremental-pass` or `fail`; a `timeout` leaves it to the next
  unit, its device possibly wedged), in that unit's worktree and under its lock, against the box's
  main-checkout fixture store. The verdict is the box's measurement readiness, not coverage: it is
  said as `<box>: box check: ready`, `ENVIRONMENT GAP (not a test failure)` with the GAP lines,
  `not at <sha>` for a ref without the script, or `could not run`, and logged to
  `logs/<stamp>-<box>-box-check.log`; no history row, unit state or run-record field reads it.
- **A GPU unit whose suite ran is followed by its backend's diagnostic legs** (gh-ocannl-1705; the
  first was gh-ocannl-1658's WMMA probe): one row per leg in `DIAGNOSTIC_LEGS` (backend, `ran` or
  `green`, `gpu` or not, log suffix, label, verdict prefix, command, what a WARNING means), each a
  `tools/` script run in the unit's worktree that ends on `<prefix>: verdict: <text>` and exits 0
  holds, 1 WARNING, 2 unavailable. HIP: `hip-wmma-probe.sh` and `vendor-idents-check.sh hip` (the
  generated routine-name table against the toolkit's headers, about ten minutes); CUDA:
  `vendor-idents-check.sh cuda` (about 40 s on rog), `cuda-tf32-parity.sh` (bench_conv_diag's seed
  parity on cifar_resnet's materialize-all step under `tf32_matmuls`, the box's main-checkout
  fixture; about 40 s) and, after a GREEN suite only, `cuda-native-ptx-rerun.sh` (the unit's own
  targets again, `@slow` too on a `--slow` sweep, under `OCANNL_CUDA_PTX_TARGET=native` and the
  unit's per-test cap, a hang said as `timeout` naming the test; in the same `_build`, so every test
  action re-executes and the next floor unit re-executes them back). The `gpu` legs (parity, native) take a
  GPU token where the unit does; the scripts that run dune go through `opam exec` and scrub the
  credential variables themselves. A ref without a leg's script is said `not at <sha>` from the
  local clone, before any transport or token. The verdict is the box's, like the box check's: said
  on the summary (`<box>/<backend>: <label>: ...`, a WARNING naming what it means), logged to
  `logs/<stamp>-<box>-<backend>-<suffix>.log`, and never in the unit's outcome, row or
  fingerprint. Each script also runs by hand, from any directory, after a toolkit upgrade. A new leg
  is a row and a script with that contract; `sweep_remote_fixture.sh` drives the rows through
  stand-ins (and the `native-on-red`, `diag-leg-without-token` mutants), `sweep_leg_scripts.sh` the
  real scripts against a fake dune. On 2026-11-02 the parity leg's first run on master said
  `STEP BROKE`: building cifar_resnet raised `Shape_error: tracing the shape scheme model: occurs
  check failed` under the default `op_shape_schemes=check` (green with it `off`).
- **The per-run record `~/.ocannl-sweep/logs/<stamp>-run.tsv` is what a consumer reads; the stdout
  summary is for humans** (gh-ocannl-977). Its absence is itself a verdict: a run that refused at
  startup swept nothing and writes no record, which is what distinguishes that exit 2 from a
  lane-stopped one. Tab-separated kind-tagged rows follow a `schema` line — **6** since the `unit`
  row gained its GPU-token readings and its window count the value `suspended` (gh-ocannl-1177,
  gh-ocannl-1039); **5** gave the `unit` row the memory model and its outcome the value `gate`
  (gh-ocannl-1035); **4** gave the
  `unit` row the window's kind (gh-ocannl-1034); **3** gave the count its fourth value
  (`vm-replaced`); **2** added the dxg fields to the `unit` row, and a
  consumer picks its parser from that number (the per-unit *state*
  files under `unit-state/` carry an unrelated schema 1 of their own) — and the sweep prints the
  record's path as a `run:` line on every exit that writes one, cancellation included — that line is
  the only locator a cancelled run gives, since it ends before the summary block. The stamp naming
  it (and every other per-run artifact) is advanced until it names nothing that exists yet —
  neither a file in the log directory nor a history row: at one-second resolution a run that ends
  inside a second releases the worktree lock inside it too, so a retry could otherwise overwrite the
  logs, fingerprints and record of the run it retries, or inherit its rows as its own units. Why a separate file rather than columns on
  `history.tsv`: history rows are per unit and append-only, so neither the run-level exit kind nor
  today's backend→box map has a unit row to live on. `test/operations/sweep_harness.sh` pins the
  record for every shape it builds — lanes, remote skips, a lane stopped mid-lane, a post-lane
  harness failure, cancellation, and the startup refusal that writes nothing. The kinds, each row's first column, with the columns
  after it in order:
  - `run`: stamp, short sha, ref, target (`<all>` when unscoped), slow flag, execution, exit kind —
    where the exit kind is `complete`, `lane-stopped` (the exit-2 shape whose recorded rows are
    real), `cancelled`, or `post-run-failed` (every unit recorded, then a harness failure in the
    post-lane phase, such as the skip aggregation, aborted the run). The kind always agrees with
    how the process exited: the record is published before those post-lane steps so a failure
    there leaves a record that explains itself, and each such failure rewrites the kind first.
  - `unit`: machine, backend, outcome or `no-row`, lane-stopped flag, log path or `-`, then the
    unit's kernel window start and end, its refusal count (gh-ocannl-979), and the window's KIND —
    `dxg` (a `-wsl` unit; the count is `vmbus_sendpacket failed` bursts), `native` (a `-linux`
    unit; the count is refused GPU queues and Xid events), or `-` (gh-ocannl-1034). The count's
    meaning is the kind's, which is why the kind is a field and why the schema is 4. That
    count has **four** permitted values, and a consumer must not validate it as numeric: `-` for a
    unit with no window (a local one, a CPU one, one whose transport has no window kind, or one that
    never ran), a number for a window that was read,
    `unavailable` where the collection itself failed, `vm-replaced` where the guest was
    destroyed and recreated mid-window, and `suspended` where a native box slept inside the window
    (the kernel's `PM: suspend entry`/`exit` lines; same boot id, so only these see it). They mean
    different things — "nobody read the box" is not
    "the bridge was fine", and only the number is a positive finding about the bridge. `vm-replaced`
    is the strongest of the four and is why the schema is 3: the window spans two boots, so its
    lines describe no single machine, and a fresh guest that came up clean would otherwise report
    the `0` that reads as "the bridge was fine" over a machine that had ceased to exist. A unit
    carrying it is environment-red, and where its outcome is one a rerun cannot reach (an `error`,
    which is what a mid-unit replacement produces) the sweep says so on its own output line. One per
    SELECTED unit, so `no-row` names a unit that should have run and whose lane never got as far as
    recording it, never one `--only` excluded. The outcome is read back out of the run's OWN
    `history.tsv` rows under the same lock that writes them, not staged beside them in a second
    per-unit channel: any second channel can disagree with the history in both directions, and no
    ordering of the two writes survives a signal landing between them, so the row is the evidence. A lane publishes a completion marker as its last act,
    so the flag covers a lane's own `die` and a relayed signal alike, and a stopped lane's earlier
    units keep their real rows (minix records hip before it stops on multidev_cc). The eleventh
    field is the MEMORY model the unit exercised (`unit_memory`): `discrete/gfx1102` (tuf/hip),
    `unified/gfx1151` (minix/hip), `discrete/sm_120` (rog-nv/cuda), `unified/apple` (metal), `-` for
    a CPU unit — declared, not probed, since KFD reports `local_mem_size 0` for both AMD GPUs on
    these kernels. It is how a consumer tells the two hip rows' coverage apart without knowing the
    fleet's hardware. The twelfth field is the unit's GPU-TOKEN readings (`gpu_tokens_summary
    field`): `<event>=<GPU holders>/<tokens>` per reading, comma-separated, `?` for one that could
    not be read, `-` for a unit with none (local, CPU, or never dialled) -- `opened=1/2,closed=0/2`,
    or `opened=2/2,refused=2/2` on a `skip` that waited out its token, the one `skip` row that
    carries a log.
  - `backend`: backend, machine. One per unit of the execution table whether or not this run
    selected it (so hip has two, minix and tuf) — staleness must be aged against the box that runs a backend TODAY, or a pre-move
    box's passes certify a path that has never run there.
- `timeout(1)` is not a portable group-killing bound, and the failure is silent in both directions.
  macOS ships none at all, which is why the repo reaches for `perl -e 'alarm N; exec @ARGV'`; and
  where one exists it is not necessarily GNU's — uutils coreutils (Rust, Ubuntu's default since
  25.10, and what BOTH GPU sweep boxes run as 0.8.0) accepts `-k` and delivers the TERM phase to the
  process group, but escalates the KILL to the DIRECT CHILD only. A descendant that ignores or
  outlives TERM is reparented and keeps running while `timeout` cheerfully reports 137. Measured
  there: `timeout -k 2 1 sh -c 'trap "" TERM; sleep 987654'` returns 137 and leaves the `sleep`
  alive. For a remote unit that means the cap says "coverage lost" while the run still holds the
  GPU and the worktree lock the next sweep must take. Both sides of `tools/sweep.sh` therefore run
  the unit under the same perl supervisor (`capped` locally, `remote_capped` emitting it as far-side
  shell text; both in `tools/sweep-supervisor.sh`): it forks, `setpgrp`s the child, and signals the GROUP on expiry, TERM then KILL, so
  the bound holds whatever `timeout` is on the far side's PATH. Exit 142 (128+SIGALRM) is that
  supervisor's expiry, on either side of the ssh (gh-ocannl-727).
- A normal sweep is deliberately incremental and records `incremental-pass`, not `pass`: that unit
  is an unknown mixture of tests executed because their dependency cone changed and tests served
  from Dune's cache. Its history row says `execution=incremental`, so it is useful evidence about
  the changed cone but must not refresh the age of full backend coverage. `tools/sweep.sh --force`
  runs `dune clean` under the per-worktree lock, then passes Dune's alias-action `--force` flag to
  both `runtest` and `@slow`; only a successful cold unit records `pass` and
  `execution=forced`. The clean is necessary because alias forcing does not reliably rerun inline
  expectations. A unit that cannot rebuild and finish inside the cap records an honest `timeout`.
  The weekly full check is therefore
  `tools/sweep.sh --slow --force`: its default unit cap is four hours, versus 90 minutes without
  `--slow`. `OCANNL_TOOL_SWEEP_CAP` overrides either; remove a scheduled Sunday override to
  inherit the shared slow budget.
  When the execution column was introduced, existing `pass` rows became `legacy-pass` with
  `execution=unknown`; old incremental evidence is retained, but cannot masquerade as a forced run.

### GPU boxes: job caps and runtime refusals

- Slow-suite budgets (gh-ocannl-2091): `SLOW_RUN_CAP_DEFAULT` in `tools/per-test-cap.sh`
  owns the four-hour budget shared by test-run batches naming slow aliases and sweep units
  launched with `--slow`. Test-run otherwise defaults to one hour, sweep to 90 minutes.
  Explicit caps win, including 0; `plan` names the cap's source. Selection precedes the
  supervisor and includes building and resolution. `tools/machine-verify.sh` gives the same
  budget to each command of a `--test`/`--record-golden` alias that `test_cap_reaches_slow`
  classifies as slow (gh-ocannl-2106); `@check`, the trip's own commands and `--run` probes
  (free text, never classified) keep its 5400 s, an explicit `--cap` bounds every command, and
  the six-hour `--trip-cap` still bounds the sum, so two slow aliases in one trip need a larger one. This portable budget also covers individual
  slow aliases; it is not a duration prediction. The former cold Metal sweep unit took about
  10.4 ks before the widen oracle joined at 612 s of action work, leaving roughly an hour
  of headroom at four hours. Parallel action times need not add to wall time. The separate
  `slow-metal_gpu_hang_failure` probe gives the OS 20 s to kill its kernel; that cost fits the
  budget but is not evidence of completed training. The per-box timing evidence is in staging#1819.
  Slow GPU batches deliberately keep no default per-test cap: a wedged device may therefore
  hold the box for the whole run cap. Set `--test-cap N` for a tighter bound; `plan` names this
  trade-off. Sweep keeps the per-test watcher on its regular leg even with `--slow`.
- **On a WSL boot, the hip unit runs its tests under `dune -j 2`, and a red minix/hip whose
  failures are all `HIP_ERROR_INVALID_DEVICE` at `hip_init`, `HIP_ERROR_NO_BINARY_FOR_GPU` at
  module load, or failed stream creation is the WSL2 dxg bridge overflowing, not the backend.**
  This bullet is about WSL boots (`minix-amd-wsl`, `rog-nv-wsl`) only; a native boot has no
  bridge, and its own limit is the next bullet. Under WSL, minix's GPU is an iGPU reached through
  `/dev/dxg`: every allocation and module load is a synchronous message
  over a Hyper-V VM bus ring, and the ring fills when the suite's test executables hold the
  device at once (dune's default there is 32 jobs). The kernel says so —
  `dmesg | grep 'misc dxg'` shows `vmbus_sendpacket failed: fffffff5` (-EAGAIN) bursts exactly
  spanning the unit — and the HIP runtime reports the lost messages under those three names,
  while `rocminfo`, a standalone `hipGetDeviceCount`/hiprtc/`hipModuleLoadData` probe, and the
  single tests all pass. The 2026-09-05 sweep recorded it twice (60+ red tests, zero test-logic
  failures) after the box had kept its VM across two host resumes, and the resumes were first
  read as the cause. **A fresh VM does not remove the refusal class.** On 2026-09-15 three freshly
  booted VMs overflowed at dune's default width within minutes (160-320 refusals, 44-85 red
  stanzas): two started by an ssh `wsl.exe` from session 0, one started by the owner's console
  WSL shell, and on both sides of a 690-line `hip_backend.ml` refactor — so neither the session
  that started the VM nor the commit decides it, the job count does. The same VMs ran the
  failing stanzas clean serially, and the console-started one then ran a cold `-j 2`
  `@runtest @train` clean in 37 minutes, both adding no refusals. So a MANUAL hip run on minix
  (`OCANNL_BACKEND=hip tools/test-run.sh run build …`) needs the cap too — and gets it: since gh-ocannl-983 `tools/test-run.sh run`/`start` injects `-j 2` and
  announces it on stderr and in the run's log whenever `/dev/dxg` is present, the batch can hold
  a GPU backend (`cuda`, `hip`) and the dune arguments carry no width of their own. An
  explicit `-j` is always honored, and only reported. Which backends a batch can hold is
  resolved, not read off `OCANNL_BACKEND` (gh-ocannl-1066; the fleet-slot bullet below says
  how): a backend set in `test/config/ocannl_config`, or a reached stanza whose marker names one,
  counts as much as the environment. Under WSL2 `/dev/kfd` and `/dev/dri` are never present and `rocm-smi` always reports
  the driver as uninitialized — neither is evidence of a lost passthrough; `hipGetDeviceCount` is.
  The cap itself is `tools/box-jobs.sh`, the single source its `box_jobs_sweep_jobs` helper and
  `tools/test-run.sh` both read, so the sweep's width and a manual run's cannot drift
  (`OCANNL_TOOL_SWEEP_JOBS=<n>` still overrides the sweep's for one run). It is applied to the
  test phase only: `test_cmd` compiles under `@check` at full width first, since the cap bounds
  GPU-holding processes, not the build. The sweep's table keys on the transport as well as the
  box and backend, read from the unit's resolved ssh destination (`-wsl` is the dxg bridge,
  `-linux` the native boot; gh-ocannl-1029). Across the bridge only minix's hip unit is capped:
  rog-nv's CUDA crossed the same bridge and a manual run there IS capped by the device probe, but
  no measurement said what width its discrete GPU tolerated under WSL, so its WSL unit stays at
  full width. The value is measured, not guessed: on the degraded bridge dune's default
  lost 67 stanzas (356 kernel-side refusals), `-j 4` still lost 27 (120), and `-j 2` ran a forced
  full `@runtest @train` unit clean in 18.5 minutes (the sweep harness pins the call shape). A
  single GPU serialises the kernels anyway, so the capped test phase is not much slower. Recovery
  for the bridge itself is `wsl --shutdown` from the Windows side, then a kick from the
  coordinator (`wake-lab.sh kick-wsl minix`) — from inside the VM the shutdown kills the session
  issuing it. A retry scoped to module loading would cover one of the three names: the refusal
  lands on `hipInit` and on stream creation too, per VM-bus message, so it would have to sit at
  the binding's error check, and it cannot help a process whose `hip_init` was refused
  (gh-ocannl-927).
- **On a native boot the limit is minix's copy-engine queues, not a bridge: the hip unit runs
  at `-j 8`; tuf's hip and rog-nv's cuda units run at their boxes' per-slot widths** (gh-ocannl-1029;
  the slot widths are the next bullet's). Each of those two ran one unit green at dune's default
  width with the box to itself, but a sweep unit shares the box with the batches holding its
  correctness slots -- rog-nv's cuda unit now as one of them, inside one of its two GPU tokens
  (gh-ocannl-1177, the sweep bullet above), the others holding none -- and a default-width unit
  beside them was never measured (on 2026-10-14
  tuf's 16-wide unit also had sixteen tests hung at once on a wedged device). The sweep reads the
  local view's constants (`box_jobs_hazard_cap`), so the two widths cannot drift, and leg 48 of
  `tools/test-test-run.sh` compares them on a faked box. The amdgpu driver's **SDMA
  queue pool is device-wide, not per process** (KFD topology: one SDMA engine, 6 allocatable
  queues; the kernel's refusal counts `8 total queues`, the driver's reserved ones included), and
  every HIP process that copies takes one, so what runs out is the number of GPU-holding processes
  on the box, however they are split across dune calls or correctness slots. At dune's default
  width (32) the native 2026-09-23 ladder lost `schedule_conv_gemm` to a ROCr assertion
  (`GpuAgent::ReleaseQueueMainScratch`), with the kernel logging `No more SDMA queue to allocate
  (8 total queues)` and `DQM create queue type 1 failed. ret -12`. `-j 8`, `-j 4` and `-j 2` logged
  no GPU kernel line at all, and `-j 8` cost 2% of the wall time (946 s against 921 s). That
  signature is what to look for in `journalctl _TRANSPORT=kernel` when a native hip unit fails
  wider than its cap — and the sweep looks for it itself: a `-linux` unit's kernel window is the
  native kind, and a refusal in it (or the ROCr assertion in its log) buys the unit its serial rerun
  (gh-ocannl-1034, below). `dmesg` is restricted on both native boxes (`kernel.dmesg_restrict=1`);
  the user reads the kernel journal through the `adm` group. rog-nv's native cuda unit ran green at
  24, 8, 4 and 2 with no NVRM/Xid line, alone: those ladder runs had the box to themselves, no other
  batch holding its GPU, which is the budget the box's two GPU tokens keep the unit inside now.
  The sweep's cap is `BOX_JOBS_SDMA_CAP` in `tools/box-jobs.sh`, one unit's width with the box to
  itself; lukstafi/ludics-lite#344 ran the unit green at `-j 16` too (1223 s, no faster). On
  tuf-amd-linux (discrete gfx1102) a benign kernel line looks like trouble to a loose grep: the
  idle dGPU runtime-suspends, and every resume re-prints its ring table (`ring sdma0 uses VM inv
  eng 12 on hub 0`), so filter that line out before matching `sdma`.
- **A manual or worker batch on a native GPU boot is capped per correctness slot, and
  `tools/test-run.sh run`/`start` injects it** (gh-ocannl-1033): `-j 4` for `OCANNL_BACKEND=hip`
  where the KFD topology (`/sys/class/kfd/kfd/topology/nodes/*/properties`) reports an SDMA pool
  no larger than minix's 6, and `-j 8` for `OCANNL_BACKEND=cuda` where `/dev/nvidiactl` exists (a
  native NVIDIA boot; a WSL boot keeps the dxg cap), and `-j 8` for hip where the pool is larger
  (tuf-amd-linux's gfx1102 reports 2 engines x 6). Each is the width the box's correctness slots
  were measured at, all of them at once (lukstafi/ludics-lite#316 and lukstafi/ludics-lite#344): minix runs four slots
  (four `-j 4` hip batches, and two `-j 8` and one `-j 16` unit, were green; only dune's default
  32 has drained the pool), tuf three (three `-j 8` hip batches green), rog four, only two of
  them GPU tokens (three or more concurrent cuda batches hit `CUDA_ERROR_OUT_OF_MEMORY` in
  `fused_classifier` in 2 of 6 rungs, at 10.0-10.2 GiB of 12 in use; two cuda beside two cc ran
  green, lukstafi/ludics-lite#391). So each hip width is a measured budget over the box's slot
  count (`BOX_JOBS_SDMA_BUDGET / BOX_JOBS_SDMA_SLOTS`, `BOX_JOBS_WIDE_SDMA_BUDGET /
  BOX_JOBS_WIDE_SDMA_SLOTS`), and the cuda one is `BOX_JOBS_NATIVE_CUDA_CAP`; the slot counts
  must agree with the fleet's `FLEET_BOX_CORRECTNESS_SLOTS` default in lukstafi/ludics-lite, and
  rog's `BOX_JOBS_NATIVE_CUDA_TOKENS` with its `FLEET_BOX_GPU_TOKENS`. On rog a CPU batch is
  capped too, at `BOX_JOBS_NATIVE_CPU_CAP` (`-j 8`, the only width cc ran at in those rungs;
  gh-ocannl-1065): four uncapped cc batches would run 96 jobs on 24 cores. That cap is fleet
  policy, not a device hazard, so it is keyed on the fleet's own name for the box —
  `FLEET_LOCAL_BOX`, else the hostname through `FLEET_HOSTNAME_MAP` (default: `rog-nv*`, `rog`),
  the rule fleet-worker.sh's `detect_local_box` applies — beside `/dev/nvidiactl`, never on the
  device alone: any other NVIDIA machine, and every other box, keeps
  CPU batches uncapped. An explicit `-j` still wins, and only says the cap exists. A batch that
  can hold several backends takes the tightest width any of them meets (hip and cc beside a small
  pool on rog: hip's `-j 4`). No stanza names hip today, so `runtest test/operations` with the
  stock cc configuration holds no HIP process on minix and stays uncapped; with hip as the backend
  `test/config/ocannl_config` sets, it gets `-j 4` like `OCANNL_BACKEND=hip`. `tools/test-run.sh plan
  <dune args>` prints the resolved backends and why, the width and the slot, and runs nothing but
  the readers' build. A batch there that holds no GPU must take its fleet slot
  as `execution slot --cpu`: the slot is fail-closed, so an undeclared cc batch waits on one of
  the two GPU tokens and rog runs at two batches, not four — which `tools/test-run.sh` now does
  itself (next bullet). Measuring a slot count: the
  dune shared cache restored nothing on these boxes (a `dune clean` + `--force` batch ran ~396
  `ocamlopt` processes, per `_build/trace.csexp`, cache on or off), so such batches are
  compile-inclusive; and without `dune clean`, `--force` does not re-run the tests at all.
  `tools/test-test-run.sh` fakes the topology (`OCANNL_TOOL_KFD_TOPOLOGY`), the device
  (`OCANNL_TOOL_NVIDIA_DEVICE`) and the fleet's name for the box (`FLEET_LOCAL_BOX`,
  `FLEET_HOSTNAME_MAP`) as it fakes the bridge. All three device probes default, harness-wide,
  to paths that do not exist (`hermetic_probes`, as `tools/test-mutation-run.sh` does); a leg
  opts in to a device and goes back through that function, never `unset` — an unset or empty
  probe falls back to the real device, and on minix the lifecycle and repeat legs' runs then
  took `-j 4` while still passing (gh-ocannl-1108). Leg 70 runs last to catch such a leak.
- **`tools/test-run.sh run`/`start` takes the fleet's run-time correctness slot itself**
  (gh-ocannl-1004), as `fleet-worker.sh execution slot --cpu|--gpu`, on any box whose deployed
  `fleet-worker.sh execution slot --probe` answers (lukstafi/ludics-lite's issue-wave skill; the
  probe also proves that version runs a nested slot inside an enclosing one, so a worker's own
  `execution slot` wrapper around the runner costs one slot, not two). The kind, and the width
  above, read ONE resolution of the backends the batch can hold (`tools/batch-backends.sh`,
  gh-ocannl-1066), made by the run's first phase (the supervisor's child, under the lock, the cap
  and the run's signal handling; gh-ocannl-1106), which rewrites the recorded command with the
  injected width before it runs dune; a `stop` or an expired cap there is an ordinary CANCELLED
  or TIMEOUT run whose log says dune was not started. `--cpu` means
  none of them holds a GPU. It asks two
  questions. First, which backends do the stanzas the run can reach NAME (`; ocannl-backend: cuda`
  and friends, which hold that backend whatever the configuration says): `ocannl_slot_kind`
  (`test/config`, over `Test_scan.Slot_kind`, reading the markers `env_var_deps` enforces over
  the closure of the argv's aliases -- dependency aliases across directories, generated
  `runtest-<name>` included, every file a stanza mentions matched by basename to any rule's
  targets, every compilation and every source-like producer -- an over-approximation with a stated
  boundary, not a proof; read from a closed set of argv shapes: `runtest` with plain relative
  directories, `build` with `@`/`@@` alias targets or plain compilation products, options from a
  harmless list; any other word -- `.`, `--`, a rule's output, `--root`, an alias-naming or
  unknown option -- is every backend) answers, so `runtest test/operations` is a GPU batch while
  `@test/operations/runtest-<cpu test>` is not; a Metal marker counts even on Linux, where its
  stanza compiles the stub. A rule target whose suffix is absent from
  `Slot_kind.data_suffixes` (for example `.err`) counts as a source-like producer, so `Slot_kind`
  takes a plain build to reach its action and `slot_kind_live` refuses a backend marker there.
  Give diagnostic outputs a data suffix such as `.log` or `.out` when compilation does not need
  them.
  Second, the configuration, which counts for every batch but two shapes
  (gh-ocannl-1227, the maintainer's decision replacing gh-ocannl-1095's proof that a batch read
  none, which needed an exact model of every dune construct and grew one per review round): every
  target on `Slot_kind.config_free_aliases` (`@test/operations/scans`, `@metal-codegen`), which
  then holds only what its stanzas name, or a plain compilation product (`build bin/x.exe`,
  `build ./test/operations/x.exe` as `bisect` spells it, a library's `.cma`/`.cmxa`), which
  compiles and runs nothing -- no backend, `--cpu`, dune's own width. Either fails closed: a
  closure that reaches a configuration reader all the same (a tree the scan below has not judged
  yet) counts the configuration; a dependency the closure cannot follow (an alias spelled
  through a pform, an `(include …)`, a `dynamic-run`) makes any batch reaching it every backend;
  and an unread stanza head anywhere (`include`, `cram`), or a reader of a configuration the
  runner does not resolve (the check below, run by `ocannl_slot_kind` too), makes every batch
  every backend. So a single `none`-marked
  test outside the list counts the configuration again: under a GPU configuration it takes a GPU
  slot, the loud direction, by design; `@lifecycle` stays off the list because its probes read
  the configuration. `slot_kind_live` (in `@test/operations/scans`, reading only the files its
  deps hand it) keeps both answers true of the tree: no listed alias's closure holds a
  configuration reader or a dependency it cannot follow; what every build reaches (compilations,
  source-like producers, their files) holds no reader and names no backend; every reader (a stanza declaring `(env_var OCANNL_BACKEND)`, or neither it nor a
  marker) runs where its nearest configuration is one the runner resolves -- in `BATCH_CONFIG_DIRS`, or copied
  from one or from the ambient copy, whose producer must keep its two-`cat` shape -- so none reads
  the gitignored root one (gh-ocannl-1249); and no `include`, `cram` or other unread head, nor a
  `dynamic-run`, is in the tree. A backend an action fixes (`setenv`, `--ocannl_backend`,
  `%{bin:env}`) is its marker's to name; nothing analyses actions. The configurations are resolved, not read off `OCANNL_BACKEND`, because an
  ordinary cc batch leaves it unset: `ocannl_read_config`
  (`test/config`, the same Utils resolution a test run makes) is built and asked from each
  directory whose `ocannl_config` sets a backend — `test/config` (copied by every `test/*`
  directory and `bin/`) and `arrayjit/test`. No backend at all is not cc (`Context.auto` then
  tries metal, cuda and hip first), and a dune argv can pick one the configs do not show
  (`exec`, whose program may choose its own, is every backend; a command-line
  `--ocannl_backend` adds its name), and anything unreadable — a build or read that fails, an
  answer outside the tool's grammar — is every backend: `--gpu`, the slot's fail-closed default,
  and the tightest width. `tools/test-test-run.sh` leg 57 fails a new tracked
  `ocannl_config` naming a backend outside those directories. The slot's wait (600 s,
  `OCANNL_TOOL_SLOT_WAIT`, at most half of `--cap`, so a refusal returns before the cap's alarm)
  comes out of the cap; a refused or unreachable slot
  is the verdict `SLOT REFUSED`, exit 75, never a test verdict. `repeat` takes no slot (wrap it
  yourself), `OCANNL_TOOL_FLEET_WORKER=none` turns it off, and a box outside the fleet runs as
  before, silently. A measurement runs as `fleet-worker.sh execution hold --request <request_id>
  -- tools/test-run.sh ...`, with no slot override: the hold exports `FLEET_MEASUREMENT_HELD`, and
  the slot then runs the batch under it, logging `EXECUTION SLOT <box>: inside measurement <id>`
  (lukstafi/ludics-lite#480), which `slot_refusal` reads as an admission like a held slot's.
- **A GPU-box batch that resolved to cc alone is green and tested nothing on the GPU, so the
  runner says so, and `--expect-backend B` refuses it** (gh-ocannl-1433). Two wave legs
  (staging#1268, staging#1269) passed on rog and minix with `OCANNL_BACKEND` unset and the test
  configuration naming cc. On a box where cuda or hip meets a width hazard (the local view above,
  so the fleet's Linux GPU boxes and never a Mac), `run`/`start`/`fresh` print `WARNING: this box
  has a GPU backend (<b>), but this batch holds only cc: ...` in the announcements, the run's log
  and the digest, and `plan` prints it too; a batch holding no backend at all (the scans) is not
  warned about. `--expect-backend B` (repeatable, before the dune arguments) states what the batch
  must hold: one that does not hold every B, or whose backends are unread (`exec`), is refused
  before the slot, the probe and dune -- verdict `BACKEND REFUSED`, exit 2, the digest naming what
  it held -- and a held expectation, cc included, silences the warning. The option makes `run`
  resolve the batch on any box, a Mac included. A GPU leg passes the backend it is for:
  `OCANNL_BACKEND=cuda tools/test-run.sh run --expect-backend cuda build @...`. Leg 88 of
  `tools/test-test-run.sh` pins both, and its warning stays clear of the words leg 49's
  width-silence controls grep for (`cap`, `sdma`, `nvidia`, `dxg`).
- **A tree carrying a backend marker `env_var_deps` refuses is `MARKER REFUSED`, not every
  backend** (the chip from gh-ocannl-1652). A `; ocannl-backend: cuda` with no reason used to make
  `Slot_kind` raise, which `ocannl_slot_kind` answered as `unknown:`, so the runner took the batch
  to hold any backend: a GPU slot and a capped width for a typo the scan reports loudly.
  `Slot_kind.marker_refusals` now answers each marker the shared contract
  (`Dune_stanza_scan.backend_marker_contract`, plus `backend_rule_of`'s second marker) refuses as
  `refused: <dune file>:<line>: <why>`, the reason worded by `Scan.backend_marker_malformed_reason`,
  the same wording `env_var_deps` fails with. Wherever the runner resolves a batch's backends (a
  fleet slot, a width hazard, `plan`, `--expect-backend`), it then exits 2 before the slot, the
  probe and dune, the digest listing each marker under `marker refused:`. Off those paths nothing
  is resolved and nothing is refused, so the scan remains the check of record. Leg 93 of
  `tools/test-test-run.sh` pins the runner's half, and `slot_kind_cases` the parser's, on the
  fixture stanza.
- **`.github/fleet-leg` makes the fleet's `execution leg` refuse a GPU leg whose command names no
  backend** (gh-ocannl-1542; lukstafi/ludics-lite#766, lukstafi/ludics-lite#773). The runner's
  warning above reaches only someone who reads the log; four wave legs (for gh-ocannl-1340, 1452,
  1432, 1456) set no backend and reported green anyway. The leg reads the file on the box at the
  commit under test and, for any leg not declaring `--cpu`, requires the runner's argv to hold
  `--expect-backend B` (before any `--`) or to lead with `env OCANNL_BACKEND=B`, else `EXECUTION LEG
  REFUSED ... pins no backend`, nothing run. So a GPU leg is `-- env OCANNL_BACKEND=cuda
  tools/test-run.sh run --expect-backend cuda build @...` (the `env` pin is what admits `bisect` and
  `repeat`, which take no `--expect-backend`); a cc leg declares `--cpu` or names cc. The pin proves
  only that a backend was NAMED: `--expect-backend` is the runner-side proof it was held. Only argv
  words count (not a `sh -c` string, not the caller's exported environment, which does not travel),
  and a line outside the grammar (`#` comments, blanks, `pin-flag <option>`, `pin-env <NAME>`)
  refuses every leg, `--cpu` ones included -- edit the file only with the grammar in ludics-lite's
  `issue-wave/references/executions.md`, *A leg on another box*, at hand.
- **Runtime-refusal signature table.** These are the exception names `tools/sweep.sh`'s
  `ENVIRONMENT_REFUSALS` treats as the environment refusing a run rather than a test judging it;
  dune prints an uncaught binding error as `Fatal error: exception <name>:` with the status on
  the next line, and the sweep keys on the name — or, for a name whose row below gives one
  qualifying status, on the name AND that status. A GPU unit's `fail` whose log carries any of them
  is *environment-red* and gets a **serial rerun** (gh-ocannl-945): every failing stanza again as
  `dune build -j 1 @<dir>/runtest-<name>` (or `@<dir>/<alias>` for an explicit rule), one dune
  call each, under the same worktree lock and the unit's own cap, appended to the unit's log. A CPU
  unit's log is not read for them (gh-ocannl-1039): a CPU unit meets no device to be refused, and
  its tests print exactly these names as fixtures -- the sweep harness does, when it fails inside
  the cc unit, which was then rerun instead of reported.
  rog-nv's CUDA shows the class is not minix-only, and that a status is not a diagnosis: the
  2026-09-13 forced unit lost exactly one stanza, `tensor_puzzles`, at its first `Context.auto`
  with `CUDA_ERROR_OUT_OF_MEMORY`, while the journal recorded a
  `vmbus_sendpacket failed: fffffff5` burst (three failed `dxgkio_create_allocation`s in the
  unit's first minute, after two `TimeSync` host resumes overnight). VRAM was not the limit: under WSL2 the card oversubscribes into host memory (a
  probe held 28.9 GB of a 12 GB RTX 5070 Ti and the test still passed beside it), 48 concurrent
  copies of the test peaked at 8.1 GB with no refusal, and alone it passes adding 155 MiB.
- **A burst is common, and which ones cost a stanza is a thing to observe, not to infer from the
  call site.** A per-boot census of rog-nv's journal on 2026-09-17 found three bursts since
  2026-09-01, not one: 10 refusals on 09-11 08:16:23–38, 3 on 09-13 09:59:23–26, 1 on 09-17
  07:09:01 — all three in the first ~90 seconds of a `cuda` unit, and only the 09-13 one cost
  anything. What the other two lost is not recorded: a message the bridge drops need not have been
  carrying a driver call this suite made, and nothing here observed where those landed. So do not
  read their harmlessness as a rule about which calls are survivable. Every call site in the table
  below is a bare call — `Cu.Module.load_data_ex` and `Cu.Stream.create` as much as
  `Cu.Context.get_primary` — with no retry or fallback around any of them in `cuda_backend.ml`, and
  the HIP column records module-load and stream-creation refusals that did abort stanzas. Read an
  earlier claim that a burst was the box's only one as an artifact of `journalctl -k`, which answers
  for the current boot alone; the census wants `journalctl -b <n>` per boot. Nor is a full-width
  unit evidence for a `box_jobs_sweep_jobs` cap on `rog-nv:cuda`: the 09-11 burst of 10 lost no stanza, and a
  forced full-scope unit at full width on 2026-09-17 (`--slow --force --only cuda`, 3321 s, ref
  `fa5116209`) passed with zero refusals in its own window. The serial rerun covers the case a cap
  would not.

| name (`Fatal error: exception <name>:`) | call site | statuses seen |
| --- | --- | --- |
| `hip_init` | `Hip.init`, backend `ensure_initialized` | minix, 2026-09-05: `HIP_ERROR_INVALID_DEVICE` (11), `HIP_ERROR_NO_DEVICE` (1) |
| `hip_module_load_data_ex` | `Hip.Module.load_data_ex`, backend `link` | minix, 2026-09-05: `HIP_ERROR_NO_BINARY_FOR_GPU` (34) |
| `hip_stream_create_with_priority` | `Hip.Stream.create`, backend `get_device` | minix, 2026-09-05: `HIP_ERROR_OUT_OF_MEMORY` (4) |
| `cu_init` | `Cu.init` | analogue by construction — rog-nv reaches its GPU through the same dxg bridge; not yet observed |
| `cu_device_primary_ctx_retain` | `Cu.Context.get_primary`, backend `get_device` | rog-nv, 2026-09-13: `CUDA_ERROR_OUT_OF_MEMORY` (1) — the only one of the box's three bursts to reach a stanza |
| `cu_module_load_data_ex` | `Cu.Module.load_data_ex` | analogue, not yet observed |
| `cu_stream_create_with_priority` | `Cu.Stream.create` | analogue, not yet observed |
| `cu_launch_kernel`, only with `CUDA_ERROR_OUT_OF_MEMORY` | `Cu.Stream.launch_kernel`, backend `link_proc` | rog-nv native, 2026-10-02: `CUDA_ERROR_OUT_OF_MEMORY` (1, `schedule_strided_1x1`, empty kernel window, green alone at the same commit) with NO other batch on the box -- rog had resumed from an 11.5 h suspend 40 s before the unit, its journal shows no ssh session but the sweep's own during it, and the registry held no reservation there -- at dune's default width (24; the unit was uncapped at that commit, gh-ocannl-1177); the status three concurrent cuda batches produced in `fused_classifier` (lukstafi/ludics-lite#316, lukstafi/ludics-lite#344); rog-nv native, 2026-10-31 (run `20261031T115955Z-276781`, two GPU tokens' batches): `schedule_strided_1x1` again, its unscheduled resnet-block reference reserving 3.3 GiB of local memory for one thread -- why that test, and the launch's back-off since: backend-memory.md (gh-ocannl-1773) |
| `hip_module_launch_kernel`, only with `HIP_ERROR_OUT_OF_MEMORY` | `Hip.Stream.launch_kernel`, backend `link_proc` | analogue of the CUDA launch, not yet observed |
- **A name is no longer the only trigger: the kernel's own evidence in the unit's window is the
  other** (gh-ocannl-979). A remote `cuda` or `hip` unit records its UTC window and, at the end,
  appends the kernel's lines from it to its log and fingerprint; a counted refusal in that window
  makes the unit environment-red exactly as a listed name does, and the serial
  rerun's `still red` / `all clean` stays the judge. **Which lines count is the unit's transport's**
  (gh-ocannl-1034): the destination the sweep reached the box through names its boot
  (`box_jobs_dest_transport`), so a `-wsl` unit reads the **dxg** window below and a `-linux` one
  the **native** window — the GPU drivers' own lines (`amdgpu`, `kfd`, `NVRM:`), counting KFD's
  `DQM create queue type <n> failed` (one per refused queue; `No more SDMA queue to allocate` is its
  reason and counts only where no DQM line was logged) and `NVRM: Xid` events, with pids normalised
  out of the signatures. Other driver lines — rog-nv's `NVRM: VM: invalid mmap context` burst from a
  driver upgrade's last minutes — are shown, not counted. The two sets are kept apart, never
  unioned, and a destination of neither kind collects no window and records `-`: before the split a
  native boot collected the dxg window, whose quiet-window fall-through lands on the restricted
  `dmesg`, so every native GPU unit recorded `unavailable` while its SDMA refusals went unread. The
  native query therefore judges the journal on the CURRENT boot's kernel entries (`-b -n 1`) rather
  than the window's, and then answers for the window whatever it holds. The userspace half of an
  SDMA refusal, ROCr's `GpuAgent::ReleaseQueueMainScratch` assertion, is keyed in the unit's log by
  `ENVIRONMENT_ASSERTIONS` in `tools/sweep.sh`, as glibc prints a failed assert. That matters because a list of names can only
  ever grow AFTER a miss — rog-nv's `cu_device_primary_ctx_retain` cost a remote session to
  attribute — and because some failures have no name to list at all: the 2026-09-15 minix runs
  produced `Command got signal SEGV`, which can never become a table row. A unit's collected evidence lives in a SIDECAR beside its log
  (`<stamp>-<machine>-<backend>.kernel-window`), and the trigger, the fingerprint and the record read
  only that: a log holds whatever the unit's tests printed, and this repository's own sweep harness
  dumps dxg fixtures on failure while running as a test action inside a sweep unit, so a local `cc`
  unit's log really can contain a complete synthetic burst block. Provenance is the file. The
  window's two bounds are both instants of the REMOTE's clock — the start read by the reachability
  probe when the unit began there, the end read at collection — because the log's timestamps are in
  that clock and no other, and these VMs resynchronise after host resumes. The filters live in
  `tools/kernel-window.sh`, sourced by the sweep and driven directly by the harness; the dxg one's two
  judgements come from that day's evidence on both boxes: `dxgkio_query_adapter_info` and
  `dxgkio_is_feature_enabled` failures are dropped **regardless of errno** (every VM boot logs
  them, at -22, -2, -11 and -1, and nvidia-smi emits the first constantly), while the burst is
  counted on `vmbus_sendpacket failed` **alone** — one lost message is reported as a triple
  (`dxgvmb_send_sync_msg`, `create_existing_sysmem`, `dxgkio_create_allocation`), so counting the
  `fffffff5` status would count it twice. Everything else in the window is kept and shown,
  counted or not: an unrecognised signature is what this exists to surface. Collected with
  `journalctl _TRANSPORT=kernel --since @<start>` — **not `journalctl -k`, which is that same match
  plus an implied `-b`** and so answers only about the CURRENT boot. Both sweep boxes keep a
  persistent journal (`/var/log/journal`), which is what lets a window span the VM dying inside it
  (minix's went away twice on 2026-09-15, and `dmesg` in the next session starts from the new boot
  and loses exactly the evidence being collected); `-k` throws that away again at the boot
  boundary, silently. Measured while building this: over a minix window covering three boots, `-k`
  returned 123 of its `vmbus_sendpacket` lines and the bare match all 365, and `-k` reported ZERO
  dxg lines for rog-nv's 2026-09-13 window, which actually holds 255 of them and that unit's
  three-message burst — the burst the issue was filed about. `dmesg -T` stays as the fallback only
  where journald keeps no kernel log. **The window's bounds are whole seconds, and cannot portably
  be finer**: `journalctl --since "@<epoch>.<frac>"` is REFUSED on both sweep boxes and returns zero
  lines rather than an error (measured 2026-09-15), so a fractional bound would silently blank the
  evidence; `dmesg --since` accepts it, on the branch neither box takes. Both bounds therefore round
  away from the neighbouring units — the start to the second after the reachability probe, the end
  to the second after collection — so a window can never inherit the burst of whatever ran beside
  it, at the price of a residual in the other direction (gh-ocannl-984, with the fallback's
  wall-clock bounds). The window and the count are also fields of
  the run record's `unit` row, so the consuming routine reads them without parsing a log. What the
  FINGERPRINT gets is only the block's stable half — which signatures appeared, and whether there
  was a burst at all — never the window instants, the kernel timestamps or the count: a fingerprint
  is compared bytewise against the previous failure's, and all three differ between two equally
  broken runs (161 bursts and 123 on minix within one hour), so carrying them would report
  `fingerprint moved` on every repeat of a standing environment red and cost the suppression that
  makes this output readable. A bridge that starts failing, stops, or fails in a NEW way still
  moves it — which is why the block carries its deduplicated signature list uncapped even though
  the raw lines it shows are capped at 40.
- **Reading a rerun's verdict.** Adding a name means adding it in both places: the table in
  `sweep.sh` gates the rerun, this one records what the name has been seen with. Qualify a name
  by its status (`<name> <STATUS>` in `ENVIRONMENT_REFUSALS`) when the call site also fails for
  the test's own reasons: a kernel launch is where device-memory pressure from a concurrent
  suite lands, but also where a kernel's own resource bug lands deterministically
  (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, or `CUDA_ERROR_INVALID_VALUE` for shared memory over the
  limit) and where an earlier kernel's sticky fault surfaces (`CUDA_ERROR_ILLEGAL_ADDRESS`), so
  only the memory status buys the launch's rerun. An `OUT_OF_MEMORY` that is the kernel's own
  (local memory too large for the resident threads) stays red at `-j 1`, which is what the rerun
  is for. A name is worth
  adding even now that the kernel evidence triggers too: it names the call site for the table. The verdict is written as `serial
  rerun:` lines in the log AND the fingerprint (outside the fingerprint's 60-entry bound, so a
  wide red cannot drop it), and quoted in the sweep's summary: `still red: <aliases>` names the
  stanzas red on their own — including one generated alias per member of a `(tests (names …))`
  stanza — read those first, they are the test-logic failures the noise was hiding; `all clean`
  says every red target passed alone; `unjudged (exit N): <aliases>` are the ones the cap cut
  short, never folded into `all clean`. A `(test)` golden diff, whose site names only its
  `.expected`, reruns as the stanza its resolved `diff --git` header's `<dir>/<name>.exe.output`
  operand names, `@<dir>/runtest-<name>` (gh-ocannl-1464); an explicit rule's `.expected` site
  stays unmapped, since nothing in the log pairs it with its rule. One or two inline-expectation sites use a capped
  `@<dir>/runtest` fallback and name it on a `directory fallback` line; three or more remain on
  `unmapped: [<site>]...`, as do unnamed spans and bare `target` rules. The bound keeps a wide red
  from becoming a full serial directory suite while giving a small inline red the same retry.
  The row's outcome stays `fail`; the rerun never changes a verdict, it explains one. Why this
  exists: the same day's wide run hid a genuine hip-only
  regression under the bridge noise (`autotune_fission_sketch`, whose staging#639 MMA-count
  claims assumed every GPU backend seeds a tensorized sketch for an f32 site — HIP's rocWMMA
  advertises no f32 triple, so the counters are zero there by design; now gated through the
  seeder's own decision, gh-ocannl-943), and the executable segfaulted after its `hip_init`
  was refused, so no fingerprint could have shown it — only rerunning the 27 red stanzas at
  `-j 1` on the box did, 4 of them staying red. The harness pins the call shape, all three
  verdict channels, and that a red without a signature gets no second run.
- Capped single-test diagnostics keep live stderr: `tools/test-run.sh run` (also `start`
  and `plan`) adds `--no-buffer` to a simple `build @DIR/runtest-NAME` (also `slow-` or
  `train-`, recursive or not), with optional `-j N`, `-jN`, `--force` or `-f`, when either
  cap is positive. The default 3600 s run cap makes this the normal single-test behavior;
  `--cap 0 --test-cap 0` disables it. The effective command is recorded. Dune otherwise holds
  action stderr until completion, losing evidence if the cap kills it (the first
  gh-ocannl-2082 Metal window). Broader batches keep each action's output contiguous for the
  person reading the log. Unfamiliar options receive no automatic flag; even `--auto-promote`
  and `--stop-on-first-error` disable injection. Pass `--no-buffer` explicitly for those
  diagnostics. A single alias may still build parallel dependencies, so use `-j 1` when output
  order matters. This changes no cap or timeout verdict. Real-Dune leg 107 of
  `tools/test-test-run-options.sh` pins both run-cap and per-test-cap stderr retention.
- **A faulted GPU does not fail the tests after it, it hangs them, so `tools/test-run.sh` caps
  each test and probes the device before a GPU batch** (gh-ocannl-1211). On 2026-10-04 tuf's
  gfx1102 took a gfxhub/CPC page fault in `bandwidth_calibration` (the test reported
  `HIP_ERROR_ILLEGAL_ADDRESS`), and every test dune started afterwards spun a core in HIP device
  init with no output: eight at once, for 32 minutes, until a person cancelled the run, with the
  run's 3600 s cap still half an hour away. A fresh `bin/device_props` spun the same way, and the
  box needed a reboot by hand. The revision was the merge that introduced
  `-amdgpu-waitcnt-forcezero`, under which HIPRTC miscompiled every `__constant__` load on gfx1102
  4 bytes off (gh-ocannl-1222, removed by staging#1005). The fault is probably that miscompile,
  but this is not proven: after the removal, `bandwidth_calibration` ran four times alone on tuf
  (the same 16 rows, golden matched, device queries clean before and after, no kernel line), and
  it had also passed once with the flag. Two bounds now apply to `run`/`start`, and since
  gh-ocannl-1252 to every other supervisor that runs GPU tests, from the same sourced helpers
  (`tools/per-test-cap.sh`, `tools/device-probe.sh`): `repeat`, each GPU unit of `tools/sweep.sh`
  and its serial rerun, and `tools/machine-verify.sh` (all below). (1) **The per-test cap**: a watcher beside dune times every child of a `dune` process, and the first one
  past the cap ENDS the run. The supervisor gets the SIGALRM the run's own cap sends, so it is the
  same reap and the same `TIMEOUT` (142), with the digest naming the test from the run's
  `test-cap` record and the remedy (`--test-cap 0`, or a larger one); only on a GPU batch does it
  also point at the kernel journal. Ending the whole run is deliberate: killing only the hung test
  would let dune start the next one onto the same wedged device. An explicit `--test-cap N` (or
  `OCANNL_TOOL_PER_TEST_CAP`) applies to any batch; the DEFAULT, 1500 s, applies only to a batch
  `run` resolves to hold a GPU backend whose targets do not reach the `slow` aliases. 1500 s sits
  above every runtest/train action in the four boxes' dune traces (the longest 1021 s
  `fsm_transformer` on minix, 810 s `autotune_smoke` on mac-studio, 757 s `circles_conv` on tuf,
  all under correctness-slot load, where load stretched tests 5-15x), but no trace holds an `@slow`
  member when that bound was chosen: `slow-cifar_conv` is box/backend dependent (2026-11-08
  sweep traces: minix HIP 155 s, rog CUDA 578 s, tuf HIP 1573 s, downloads excluded), so it
  can pass 1500 s legitimately, and CPU batches are not the failure class. So `@slow` and CPU
  batches keep only the run's cap unless asked; `plan` says which applies and why. The default and
  the watcher live in `tools/per-test-cap.sh`, which `tools/sweep.sh` sources as well: a sweep unit
  on a non-CPU backend runs the same watcher beside its `@runtest @train` leg (not its `@slow`
  leg), and beside each stanza of its serial rerun and its completion pass (unless they reach
  `@slow`), signalling the leg's own supervisor, so a hung test ends the unit as `timeout`, or the
  rerun with the stanzas it did not reach `unjudged`, and the lane's line names it
  (`OCANNL_TOOL_SWEEP_TEST_CAP`, 0 lifts it). The serial rerun needed it most: what triggers it is a
  kernel-side refusal signature, when the device is most likely faulted. `repeat` runs the watcher
  beside each iteration's dune, its default decided by `run`'s own resolution of the batch, and a
  test past it ends the whole set; `tools/machine-verify.sh` runs it beside each `--test` and
  `--record-golden` alias of a GPU trip (`--test-cap N` for any operation, `--run` included; 0
  lifts it), the trip ending with 142. Before that, the 2026-10-14
  sweep (stamp 20261014T051751Z) had sixteen tuf hip tests hung at once burn the unit's full
  5400 s cap with no test named. Tests are found as children of `dune`, because dune
  starts each action as its own process-group leader. The run's group kill therefore reaches dune
  and not the action, but dune's TERM handling kills every action it has running, a
  TERM-ignoring one included (measured on 3.24). A watcher counting every group in the run would
  also time the fleet slot's sleep inhibitor, which lives for the whole batch. The watcher reads
  `ps` (procps or BSD; leg 75 runs wherever a `$0`-renamed process reads back as `dune`, macOS
  included), and records `test-cap-off` where `ps` cannot list processes (Git Bash). It times
  each test on its OWN clock, not by `ps etime` (gh-ocannl-1319): minix's procps, minutes after a
  resume from suspend, reported a test a minute old as 38109073018720 s old (etime is derived from
  the boot time, which the resume skewed), and the cap ended the run. A test is born where the
  watcher first lists it, less its etime clamped to the time since the previous listing (which
  did not show it), so a skewed etime moves a birth by at most one poll, and an exec that renames
  it (a wrapper handing over) does not restart its age, only a new pid, parent or start token does
  (gh-ocannl-1448: a process given the test's pid under the same dune within one poll would
  otherwise inherit its age); the clock stops while
  the host sleeps (`CLOCK_UPTIME_RAW` on macOS, `CLOCK_MONOTONIC` on Linux), so a suspend never
  counts against a test either. The start token is `/proc/<pid>/stat`'s starttime on Linux (field
  22, kernel clock ticks since boot, not procps's boot-time-derived wall time, which the same
  resume skews) and `ps -o lstart=` elsewhere; an unreadable one decides nothing. Leg 75 serves the
  watcher a stub `ps` with that etime, and times a test that execs itself under ten names; leg 91
  advances a test's starttime through the watcher's `/proc` seam (`OCANNL_TOOL_TEST_CAP_PROC`) and
  it is never capped, while a steady one is. (2) **The
  device probe**: a batch `run` resolves to hold a GPU backend queries each such backend once,
  through `bin/device_props --ocannl_backend=<b>` from `test/config`. Build and query both run
  inside the fleet slot, as the slot's command (`_probe`, which then execs dune): a measurement
  hold sees neither a compile nor device traffic from a batch it refuses, and the slot's wait
  stays the only step between `clamp_slot_wait` and the slot, so a busy slot still reports SLOT
  REFUSED before the cap can report TIMEOUT. There is no real-exe probe on macOS: a freshly
  linked `device_props` can sit in dlopen for minutes under XProtect (runs over ssh or launchd
  are not exempt), which would read as a hung device; metal batches keep the per-test cap. A query that does not answer within
  `OCANNL_TOOL_DEVICE_PROBE_CAP` (60 s; a healthy one takes 0.2-0.3 s) refuses the batch as
  `DEVICE UNHEALTHY` (exit 69) without starting dune. A query that fails fast (cuda or metal
  asked on an AMD box, as every unread batch asks) is recorded in `device-probe` and passed over:
  its tests fail fast too rather than spin. A batch whose backends were never resolved (no width
  hazard and no fleet slot) is not probed, and gets no default per-test cap. The query and the
  per-backend loop are `tools/device-probe.sh` (POSIX sh), shared since gh-ocannl-1252: `repeat`
  probes before every iteration (its `_probe`, inside the iteration), and a refusal ends the set; a
  GPU unit of `tools/sweep.sh` probes after its compile-only prebuild and before any test leg, and
  again before its serial rerun, a refusal filing the unit as `error` with a `DEVICE UNHEALTHY,
  dune not started` line (rerun: `serial rerun: DEVICE UNHEALTHY`, every stanza unjudged) rather
  than at the first test's cap (`OCANNL_TOOL_SWEEP_DEVICE_PROBE=none` turns it off, a path names a
  stand-in, `_CAP` bounds it); and `tools/machine-verify.sh` probes the pinned GPU backend before
  its first device use, a refusal ending the trip with 69 (`--probe-cap`). None of them probes a
  local unit on macOS without a stand-in, for the XProtect reason above. Both bounds are deliberately outside the driver: the
  recommendation on the issue preferred them to a device-health watchdog inside the batch, which a
  faulted driver can hang along with everything else. On a 69, or a per-test 142 on a GPU
  batch, stop GPU work on that box and read `journalctl -k` before retrying. Legs 75-77 of
  `tools/test-test-run.sh` pin the cap, its controls (a slow dune, the slot's holder), the probe
  and `plan`'s report, and leg 83 both bounds under `repeat`; `test/operations/sweep_harness.sh`
  pins them in a sweep unit, its serial rerun and its completion pass (a local and a remote unit),
  and `tools/test-machine-verify.sh` in a trip (`test-cap-*`, `probe-*` cases; a fake `uname`
  stands for a Linux far side).
- **Inside every process, a GPU backend's device discovery and device opening have a deadline
  too** (`device_init_deadline`, 120 s, gh-ocannl-1281): `Backends.With_device_init_deadline`
  guards every call of the cuda, hip and metal entry points that ask the driver about devices or
  open one, serialized process-wide, and past the bound a watchdog thread of `utils_stubs.c`
  writes one line opening `OCANNL device_init_deadline:` and kills the process by SIGALRM (142),
  reaching bare `dune`, `bin/` tools and user programs the harness bounds never see. A thread,
  not a SIGALRM handler on the real-time timer: seven review rounds of staging#1244 hardened a handler
  (signal masks, pending and external signals, async-signal safety, a shared timer) before the
  thread removed the whole class. Under dune that death is a red action whose signal the default
  display does not name, so the digest reads the line itself, at a line's start, in the whole log,
  as `WEDGED DEVICE` (leg 87). A program with its own SIGALRM handler gets no deadline, and none
  exists on Windows; `test/operations/device_init_deadline` pins all of it through a test stall
  hook. "Its own handler" is read from the handler slot alone, never `sa_flags`: macOS keeps a
  caught signal's `SA_SIGINFO` across `exec`, so under GitHub's macOS runner agent every
  descendant reads `SIG_DFL` with `SA_SIGINFO` set, and a flag check silently disarmed the deadline
  there while bare-metal Macs passed. Every bound in it is measured on `CLOCK_MONOTONIC`, never
  counted in naps: on that loaded VM 200 "10 ms" naps took 10 s (`taskpolicy -b`, or the test's
  in-process throttle, reproduces it on a bare-metal Mac; `taskpolicy -b` reaches every descendant,
  so `taskpolicy -b tools/test-run.sh fresh @test/operations/runtest-device_init_deadline` reruns
  the whole test that way, and passes). Each arm is a generation, and a region's exit disarms only its
  own: a region a `fork` child inherited would otherwise cancel a deadline a new domain of the child
  armed (gh-ocannl-1446, the `fork-disarm` case). The disposition is read at arm time, so a handler
  installed mid-region is overridden only at the death (a recorded boundary). HIP's `hipInit` takes
  the deadline itself, inside `ensure_initialized`, because `Impl.hip_to_code` forces it ahead of its
  guarded discovery; `Utils.driver_init_test_stall` stalls there
  (`arrayjit/test/test_hip_driver_init_deadline`, a HIP leg only).
- **Revisit a per-action bound from the traces, through `tools/action-durations.sh`**: it reads a
  dune trace (`_build/trace.csexp` by default — a bare dune's; a `tools/test-run.sh run` writes its
  batch's into the run directory and keeps only the five slowest, below) and
  prints the slowest processes dune started, each named by its alias or first target relative to
  the trace's own `build_dir`, or with `--group prog|dir` the max/total per test executable or
  directory; `--prog '\.exe$'` keeps the tests. A process the trace saw start and not finish
  prints `open`, its seconds a lower bound. Every `tools/test-run.sh run`/`start` already keeps
  the five slowest of its batch as `slowest` in the run directory, and the digest prints them —
  a `TIMEOUT`'s too, where they show how close the rest came — so the data piles up under
  `~/.ocannl-test-runs` with no manual step. The batch's dune gets `--trace-file=<run dir>/trace.csexp`
  (after the recorded `cmd`, for `build`/`runtest`/`test`/`exec` only, never when the argv names its own
  trace file), so the batch neither reads nor writes the worktree's `_build/trace.csexp` (the readers'
  and device-probe builds still rewrite it, as any dune command does, under dune's own build lock):
  deleting that shared file instead, outside the build lock, cost a concurrent manual `dune build`
  its live trace (review of staging#1079). A reader that fails (a dune older than 3.24 writes a trace
  format it refuses) or times out leaves `slowest-skipped`, and the digest prints its reason. The supervisor reads the run's trace as the
  verdict's last step, on every exit path, the caps' included, and deletes it after; the reader holds
  no descriptor past stdio, is KILLed past its 30 s and abandoned if even KILL does not reap it, so a
  stuck read never withholds the verdict or the worktree lock. Leg 80 of `tools/test-test-run.sh`
  pins it. Never read a trace by hand to answer this: it also
  records dune's environment, argv and each process's arguments and captured stdout/stderr, which
  can hold credentials (gh-ocannl-1280). The tool decodes only event kinds, digit-only time slots,
  `config init`'s `build_dir` and five process fields, skipping the rest by length prefix, and its
  refusals quote offsets, never bytes. That boundary assumes producer-framed csexp: a tampered
  length prefix can swallow the environment into a whitelisted field, still well-formed, so
  protection against tampered traces is producer-side scrubbing (ahrefs/ocannl#1280), not the
  reader. `tools/test-action-durations.sh` seeds a canary in every non-read field of the schema it
  inventories, rejects one decode-and-print twin per such field by the canary alone, and pins the
  tampered-prefix leak as the documented boundary. `tools/ci-shard.sh weigh` holds the same
  boundary over its own fields (gh-ocannl-1287): it opens only `process finish` and decodes `prog`,
  `dir`, `target_files`, the time slot's digit-only duration and, of `rusage`, only the digit-only
  `user_cpu_time` and `system_cpu_time`; `tools/test-ci-shard.sh` seeds and twins its inventory
  the same way. Each
  reader keeps its own whitelist rather than share a module, since what each may decode differs
  and each harness rewrites its own subject's whitelist line to build its twins.

### Skip coverage

- A forced full-suite sweep also intersects the backend-scoped `Verdict.skipped`
  executable-and-claim keys from every successful unit through `tools/aggregate-skips.sh`
  (gh-ocannl-792), writing
  `logs/<stamp>-skip-coverage.txt`. What it intersects is each unit's **per-action verdict
  records**, never its log (gh-ocannl-1114): the sweep exports `OCANNL_TOOL_VERDICT_RECORDS` (a
  directory beside the worktree, `<wt>.verdict-records`, emptied when the unit starts) into the
  suite, the serial rerun and the completion pass alike, and Verdict writes one file per action —
  keyed by working directory, executable and arguments, so one executable under two stanzas is
  two actions — truncated when the process starts. Each file therefore holds its action's final
  attempt by construction: a retry rewrites its own action's records, an action never re-run keeps
  its own, and nothing depends on attributing lines of a merged stderr stream (five review rounds
  of lukstafi/ocannl-staging#857 went to that inference, and dune truncating a noisy HIP action's
  output once dropped records from the log outright). The sweep collects the files, concatenated,
  into `logs/<stamp>-<machine>-<backend>.verdict-records`; a unit whose records cannot be read or
  hold no `OCANNL_TOOL_VERDICT_ACTION` header — a swept commit predating them — is reported `skip
  evidence unavailable` and not counted, since its silence would read as every claim executed.
  The stderr copy of each record stays in the log for humans. Only the action's own process
  writes: Verdict empties the variable in its own environment once it has read it, so no child —
  however spawned — writes into, or truncates, the action's file (a child is the parent's
  fixture, and `cc_march_census`'s emit children relaunch its executable from the directory of
  the action that spawned them). A process that is not Verdict-linked must unset it before running a
  Verdict-linked fixture (`sweep_harness.sh`, beside its `OCANNL_BACKEND` neutralization). A record kind the
  aggregator does not judge (`OCANNL_TOOL_VERDICT_<KIND>`, the room left for gh-ocannl-996's bypass
  record) passes through it. Beside its skips, each action's file names every claim it evaluated
  and held (`OCANNL_TOOL_VERDICT_EVALUATED`, file-only, never echoed on stderr), and a skip an
  `EVALUATED` record of the same unit names does not count for that unit: one executable under
  two stanzas — a default run skipping its tf32 leg on CUDA, its `-tf32` rule evaluating it — is
  two actions on one box, and before that record the concatenated files read as the box never
  evaluating the claim (the 2026-11-01 forced sweep's `tf32_reference_rounding` and
  `bench_seed_parity_smoke` findings). Records from before it carry none and judge as before.
  Incremental runs are refused because a cached Dune action does not run, and failed or
  interrupted units are refused because they may not have reached every test — except a red whose
  serial rerun reported `all clean` with nothing unjudged or unmapped, followed by `suite
  completed`: an incremental `-j 1` pass over the unit's whole suite (no `--force`), which runs
  exactly the actions that never completed — the dependents a red prerequisite held back, which
  rerunning the red stanzas alone does not reach. Every action has then completed and written its
  records, and the report names the unit on a `red units counted after a clean serial rerun:` line.
  Dropping such units lost minix/hip's evaluations on 2026-09-27 (red only from a ROCr scratch
  assertion at parallel width) and reported its hip-only claims as skipped on every box. An
  intersection over only the completed backends is a loud `POTENTIAL` report; it becomes `FAIL`
  only when every backend in the sweep's own unit vocabulary completed, while the sweep itself
  still exits zero so later units and their history rows are never suppressed.
  A skip caused by a host or configuration capability rather than the selected backend (a compiler
  target, preprocessing flag or filesystem feature) uses
  ``Verdict.skipped ~aggregation:`Environment``: its human stderr line stays visible, while the
  record's scope keeps it out of the backend intersection. A skip whose occurrence varies with the
  run's own configuration, with no golden row of its claim to print (which flips a stated key
  outranks), uses `Verdict.announce_skip` instead: the same stderr line and record, no stdout row,
  so the golden does not depend on the ambient environment (gh-ocannl-1819); a rule forcing such a
  skip on every box records it `Outside_sweep`, or the intersection reads the fleet as never
  covering a claim the plain run covers. The sweep's stdout
  summary quotes the report's `result:` verdict and each `FAIL:`/`POTENTIAL:` finding line (indented, under the
  `skip coverage:` line that carries the report path), so consumers of sweep output — the daily
  scheduled routine's report and notification foremost — see zero-coverage findings without
  opening the report file; the routine diffs the latest report's finding set against the previous
  `*-skip-coverage.txt` and treats `FAIL`, or a changed finding set, as notify-worthy.
  The same report intersects environment-scoped records across the successful units and judges
  completeness by BOX, not backend: the canonical box set comes from the exact swept commit's
  `# measurement-boxes:` declaration in `benchmarks/fixtures/DIGESTS.txt`, read through
  `fixture_digest.py`'s parser, while each sweep unit supplies its stable box key. The launcher must
  set `OCANNL_TOOL_SWEEP_LOCAL_BOX` to the local host's declared ID (`m4-max` on the scheduled
  machine); hostnames and CPU models are not stable fleet identities, so an unset or undeclared
  value is refused before any unit can write a mislabeled row. One successful
  forced unit is enough to represent a box because an environment-scoped gate is, by contract,
  independent of backend; when a box contributes several units, absence from any one proves the leg
  executed there. Several boxes may contribute the same backend; backend completeness counts that
  backend once, while environment completeness still counts both boxes. A claim present in every
  completed declared-box run becomes `FAIL` only when every declared box contributed; with missing
  boxes it is `POTENTIAL`, and a pre-declaration historical ref is left explicitly unaggregated. A
  claim may be backend-gated in one run and environment-gated in another (the default-policy
  `autotune_mma_companion` leg is the exemplar). An environment record in any run
  assigns that executable-and-claim key to the environment dimension; backend or environment skip
  records for the same key then both mean their box did not execute it. This ownership-before-
  intersection order prevents a different scope from masquerading as execution. A unit on a box
  outside the declaration (tuf, or a current box a historical target predates) is evidence without
  obligation: its execution clears an environment claim, but its absence never makes the matrix
  incomplete and it never stands in for a declared box; the report lists it on an
  `undeclared boxes` line. In the other direction, a declared box with no runnable unit
  is a harness refusal rather than permanent silent non-coverage.
  A configuration matrix that runs outside the fleet sweep uses
  ``Verdict.skipped ~aggregation:`Outside_sweep``: the announcement and machine-record validation
  remain, but neither sweep intersection claims ownership. This is deliberately narrow: the
  `cc_backend_trace_name` claim is executed by the Ubuntu compiler-trace CI leg at trace level 3,
  while the ordinary sweep's default configuration cannot execute it (gh-ocannl-885).

### Sweep units and verdicts

- A `git fetch` under git 2.55 (CI's ubuntu-latest, mac-studio; not the fleet's 2.53) starts a
  detached `git maintenance run --auto` whose worktree-prune task runs whenever a stale worktree
  entry exists, and a prune beside a `git worktree add` deletes the entry the add has created and
  not yet locked: the add dies on `could not open '.git/worktrees/<name>/locked'`
  (gh-ocannl-1532, which surfaced as a sweep_harness control flaking under CI load). A script that
  fetches and then adds a worktree passes `-c maintenance.auto=false` to the fetch, as
  `tools/sweep.sh` and `tools/machine-verify-far.sh` do; the sweep's `prep_cmd` also retries the
  add once after a prune, for a pruner it did not start (another session's fetch in a shared clone).
- Every test action a sweep unit runs inherits `OCANNL_BACKEND=<that unit's backend>`. The unit is
  spelled `OCANNL_BACKEND=<backend> opam exec -- dune build @runtest @train`, and Dune hands its own
  environment to the actions it runs whether or not a stanza declares the variable — `(env_var …)`
  buys dependency tracking, never insulation. A test that CONSTRUCTS a child environment — a fixture
  driving a script, a harness nesting a tool — must therefore neutralize what the launcher exported
  rather than only adding to it; otherwise it reads the launcher's backend and is green everywhere
  except inside the sweep, the one place its verdict gates anything. `test/operations/sweep_harness.sh`
  is the worked example (gh-ocannl-893): the nested sweep's environment is built with
  `env -u OCANNL_BACKEND -u OCANNL_TOOL_SWEEP_CAP -u OCANNL_TOOL_SWEEP_CONTEXT_CAP` (every
  sweep knob added since is pinned there explicitly, `OCANNL_TOOL_SWEEP_JOBS` included), and one
  aggregation is re-run with a hostile ambient backend so the neutralization cannot lapse unnoticed.
  Reproduce the condition on any alias with `OCANNL_BACKEND=<backend> dune build --force @<alias>`;
  a plain local run never sets it, which is what makes this class of failure look like flakiness.
- An unreachable machine records `skip (unreachable)`, and a sweep of skips is not a failure. It is
  not the expected steady state either: both GPU boxes are cabled and Wake-on-LAN armed, and wake
  over Ethernet from sleep and from full shutdown alike, so a run that is meant to cover CUDA or HIP
  wakes them first rather than waiting for a day someone left them on. Waking one is not the same as
  reaching it: on a WSL boot the sweep addresses the `-wsl` aliases, and WSL starts on demand or at
  login, never at boot, so a woken box answers on its Windows alias while `rog-nv-wsl`/`minix-amd-wsl`
  are still refused (a native-Ubuntu boot answers at `-linux` once sshd is up, and none of the WSL
  holder business below applies to it). Kick it with `ssh <box>-win 'wsl.exe -d Ubuntu -e true'` and re-probe for a minute or two
  while tailscaled registers. On a box woken from power-down (as opposed to resuming from
  sleep/hibernate with the owner's interactive WSL shell still open, in which case the VM survives
  the resume and none of this applies), a kicked VM can also terminate again within seconds to
  minutes — `-win` up, `-wsl` refused again. **A session running INSIDE the VM does not keep it
  alive; a `wsl.exe` process on the Windows side does.** The kick's `wsl.exe` returns at once, and
  the VM then shuts down in an orderly way (its journal ends in `systemd-poweroff`, not a panic)
  under whatever ssh sessions are running in it: 2026-09-08 lost three minix launches mid-fetch
  and mid-build that way, and on 2026-09-15 the sweep's recovery rerun lost its hip unit 76 s in,
  the VM powering off 18 s after boot. For an unattended run, hold the VM from the Windows side until
  the work on that box is over —
  `ssh -o ServerAliveInterval=15 <box>-win 'wsl.exe -d Ubuntu -e sleep infinity'` in the
  background, killed explicitly once the box's last unit (and its diagnostics) has finished. Do not
  size a fixed `sleep N` to the expected run: a lane runs several capped units back to back, with
  preparation outside the caps, and a holder that expires first shuts the VM down under whatever
  unit remains. The owner's console WSL shell plays the same role, and normally does. A Windows Update restart takes that shell away with no notice (it
  logs the console session off), so the first sweep after a patch reboot runs against an unheld
  VM. Windows schedules those restarts outside its **active hours**, which on both boxes had
  been 10:00 to midnight or 01:00, leaving the early-morning sweep exposed: the 2026-09-15 cumulative update rebooted minix at
  05:29 UTC, 21 minutes into its hip unit, as the two previous cumulative updates had inside the
  same slot. Both boxes now set `HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings`
  `ActiveHoursStart=6`, `ActiveHoursEnd=0` (the 18-hour maximum, confining restarts to 00:00–06:00
  local); `System` event 1074 from `MoUsoCoreWorker.exe`/`TrustedInstaller.exe` is the signature
  of an update restart when a unit dies with `Operation timed out`. What IS a failure is silent
  non-coverage: track the age of the last `pass` per backend, because nothing else in the project tests CUDA or HIP at all — and read a long
  skip streak as a decision nobody made, not as coverage that was unavailable.
- Report changes in the failure set, not the presence of failures. A backend's suite goes red in
  bursts and comes back (Metal's `test/operations` was red for a stretch, green again after
  gh-ocannl-632), so a sweep that shouts on every red is one that gets ignored inside a week;
  `sweep.sh` writes a sorted `.fingerprint` next to each non-pass log precisely so the previous
  run's can be diffed against it. Two properties are what make that diff mean what the consumer
  reads it as, and both are easy to lose:
  - **A fingerprint entry is a stable IDENTIFIER, not a source coordinate.** A location inside a
    `dune` file is reduced to the stanza it names (`File "test/operations/dune", alias
    runtest-foo`), because line numbers there shift under any edit to that file and a diff keyed on
    them reports wholesale change whenever an unrelated stanza is inserted above — overstating
    exactly what it is asked to measure. The identifier is not reliably a bare word on its
    keyword's line: it can be quoted, wrapped so `(targets` ends one line and its first target
    begins the next, or nested as `(alias (name slow))`. A same-line reading of the bare form
    handles none of those and falls back to the span *silently*, which looks like it works.
    Dune spells a location two ways, `line N` for a
    single-line diagnostic and `lines N-M` for a span, and a failing explicit-rule test — the whole
    scanning/golden-diff family here — produces the latter, so a selector must accept both.
  - **A non-pass with nothing extracted is its own condition, not a fingerprint of zero failures.**
    Empty compares equal to empty, so a unit whose diagnostics the selectors cannot reach is
    otherwise filed as "unchanged since the last sweep" and reported to nobody — the one failure
    shape a diffing consumer cannot see. It gets a sentinel line in the file and a line in the
    sweep's own summary, the channel the scheduled routine quotes.
- The comparison cursor lives under `~/.ocannl-sweep/unit-state`, keyed by machine, backend,
  target and slow scope. It retains the immediately previous judged verdict (skips, errors and
  timeouts do not erase it) and the previous failing
  fingerprint across intervening greens, plus each failing golden's last-touch commit. A new red
  after green, or after one of those golden commits moves, is labeled `REGRESSION OR FIX DID NOT
  TAKE`; a changed fingerprint is reported against the previous failure rather than merely the
  previous run. A red after a green also gets the ready bisect on the line after it
  (gh-ocannl-1375): `<box>/<backend>: bisect on <host>: OCANNL_BACKEND=<backend> tools/test-run.sh
  bisect <green sha> <this sha> <aliases>`, the aliases being exactly the failing stanzas the
  serial rerun maps (`rerun_aliases`, `(test)` golden diffs included), with any site no stanza is
  found for listed as left out. A green at the same commit prints `bisect: none` instead: nothing in the code moved. Standing identical reds stay quiet, which is the signal-to-noise property the
  cursor exists to preserve. The cursor key includes the requested logical ref, so an exploratory
  branch or historical run cannot become `origin/master`'s predecessor. Golden paths come from the
  full log: ordinary test diffs name their `.expected` file directly, while an explicit rule's
  resolved `diff --git` header proves its diff actually ran and supplies the first operand. Thus a
  producer crash before a later `diff` records no golden, `%{read:...}` substitutions are already
  expanded, and PPX `*_expected.ml` operands need no naming special case. Inline `ppx_expect`
  source-to-`.corrected` headers contribute their checked-in source first operand. The normalized
  fingerprint deliberately no longer carries enough source relationship to recover this provenance.
- The per-machine worktrees are reused, not recreated, so a sweep is incremental against an
  existing `_build` — seconds rather than minutes when little changed. That is what makes a daily
  cadence affordable. `--force` is the explicit from-scratch unit; a fresh CI run is the other
  path to a clean compilation check.

### Bisecting a regression

When a sweep unit (or any run) goes red and an earlier run of the same targets on the same box was
green, the two commits bound a bisect: `tools/test-run.sh bisect GOOD BAD <targets…>`
(gh-ocannl-1225; a sweep unit's summary prints it ready to paste, gh-ocannl-1375), for example `OCANNL_BACKEND=cuda tools/test-run.sh bisect --build
./test/operations/test_vec_simple.exe <good-sha> <bad-sha> @test/operations/runtest-test_vec_simple`.
Finding gh-ocannl-1222's cause took a hand-rolled `git bisect run` on tuf, and its step script had
three easy ways to lie; the verb encodes each once (tools/bisect-steps.sh says how):

- **Exit mapping.** `run` exits 1 for a red test and for a red build alike. Each step first runs
  `build @check` plus every `--build TARGET` (name the tests' executables, so a link failure counts
  too) and only then `build <targets…>`: a red build is git's skip (125), never bad, and so is a
  test run that is `BUILD FAILED` (65; a build step of the targets' own failed). A verdict that
  is not the suite's — `SLOT REFUSED` (75), `DEVICE UNHEALTHY` (69), a cap's TIMEOUT (142), an
  `INVOCATION REFUSED`, a cancellation, a run that died without a verdict or whose recorded verdict
  or commit disagrees with its launcher — makes the step exit 128, which stops `git bisect run`;
  read as bad, any of them would silently steer it. A TIMEOUT aborts too: the step cannot tell a
  hang from a wedged device.
- **Backend.** `OCANNL_BACKEND` is required (`cc` for a CPU bisect) and passed to every run of every
  step, so a configuration a commit changes cannot move the bisect to another backend half-way.
- **Isolation.** The bisect gets a detached worktree of its own under
  `~/.ocannl-test-runs/bisect-<stamp>-<pid>/tree`, never the caller's checkout, the sweep's or
  another session's, and its runs take that tree's lock; a step hands git its verdict, and git
  moves the tree, only once nothing those runs started holds it. Every step is judged by the runner frozen
  beside it at the start (`runner/`, pointed at the tree through `OCANNL_TOOL_TESTRUN_TREE`, which
  only `run` and `plan` accept), not by the tested commit's own copy, which may predate the slot or
  the caps, nor by the caller's, which may be edited meanwhile. Each step is two plain `run`s, so
  the fleet slot, the width cap, the per-test cap and the device probe apply exactly as to a
  hand-launched run; `--cap`/`--test-cap` go to every one of them.

It is first-parent by default (each step a merged state CI saw; `--no-first-parent` walks into the
merged branches), so GOOD must then lie on BAD's first-parent chain (a GOOD merged in through a
second parent is refused), and it runs both ends before bisecting: BAD must fail and GOOD must pass on this box,
or there is nothing to find. Exit 0 names the first bad commit; 1 means the range does not bisect
(an end does not hold here, or only skipped commits are left, listed); 2 a refusal before any step;
an abort exits with the aborting run's status (126 when it gave none); 129/130/143 cancelled. The
records outlive the tree, which is removed at the end: `steps.tsv` (each step's commit, verdict and
two run directories), each run's digest and log under `steps/`, git's `bisect-log` and `result`.
A bisect lasts as many steps as the range has halvings, each a build and a test run, so launch it
like a long suite: through the harness's background mode (`bg-run.sh` on the fleet). It runs where
it is invoked, with no remote dispatch of its own: for a box-specific regression, invoke it on that
box.

## Floats and measurements in goldens

- A golden line printed at a FIXED decimal precision is not made portable by lowering the
  precision: it only moves the boundary. `cifar_conv`'s epoch-30 mean loss sat at ~1.05, so its
  `%.1f` print — introduced to absorb reduction-order drift — read `1.0` on cc and `1.1` on cuda at
  the same commit, and no promotion could serve both backends. The fix that holds is the one the
  test metrics already used (`9defc92f`): exact digits to stderr, where dune does not diff them, and
  on stdout a `Verdict` claim about the property the trajectory was there to show (the loss fell
  from the first logged epoch to the last). The `@slow` goldens get this treatment the day they
  flip, because CI never runs `@slow` — only `tools/sweep.sh --slow` (the Sunday sweep) and hand runs do — so a knife-edge
  there stays hidden until a GPU run lands on the other side of it.
- gh-ocannl-725 swept that genre out of `test/training` and `test/gpt2` rather than waiting for the
  next flip, and the audit produced a rule worth reusing whenever a training golden gains a number.
  A float may sit in a stdout golden only when its value is EXACT by construction: a threshold
  constant (`moons_demo`'s convergence epsilon), a power-of-two loss scale (`mixed_prec_parity`), a
  host-side schedule evaluated in closed form (`loop_utils`' LR table and z-scores), or a small sum
  of exactly representable dyadic terms (`data_parallel`'s 67.5 / 11.25 batch losses). Anything a
  device reduction produced — a trained loss, an epoch mean, a probability, a stepped parameter —
  goes to stderr tagged `(not part of the golden)`, and stdout gets a `Verdict` claim about the
  property the number was showing. The claim must still DISCRIMINATE: a threshold the trained value
  clears and an untrained one does not, a fall between the first and last logged epoch, an argmax or
  a ranking (`mlp_bn_names`' top-3 next characters — their 0.07–0.11 probability gaps are hundreds
  of times the cross-build drift that the two printed decimals were one tie away from), or an
  absolute closed-form value within a tolerance. "Is finite" is not by itself such a claim; pair it
  with one that a wrong number fails. Nothing lints this — a `%.Nf` scan's false-positive rate on
  constants and dataset sizes did not justify another exemption list — so it is an audit rule for
  whoever adds the next training golden.
- Two details of that conversion are what the digits were quietly doing, and both are easy to drop
  (Codex found all seven instances of them in one round on staging#447). First: a pinned number is a
  TWO-SIDED constraint, and the claim that replaces it usually is not. Every loss bound in the
  makemore and conv tests was an upper bound, so a dropped negation or a backend sign error — a
  finite NEGATIVE cross-entropy — cleared all of them while the trajectory still fell; cross-entropy
  is `-log p >= 0` by construction, so each of those tests now claims validity over the whole
  trajectory beside the thresholds. The same applies to a claim about SHAPE: `mlp_bn_names`' top-3
  ranking and separation both hold for a head of 0.70 / 0.20 / 0.08, so coarse magnitude bands went
  in alongside them. Ask what range the removed digits excluded, and claim that range. Second: mark
  every relocated line `(not part of the golden)` — stderr and stdout interleave in a terminal, and
  without the tag a reader cannot tell an informational number from an asserted one.
- A training convergence threshold reads a WINDOW, not the last epoch alone (gh-ocannl-854): use
  the mean of the last ten logged epochs through `test/training/training_golden.ml`, keep the exact
  statistic on stderr, and put only its two-sided `Verdict` contract in the golden. Size the bound
  from the cross-backend/day spread of that SAME window statistic, not from one backend's final
  draw. `circles_conv` is the motivating control: its 2026-08-29 multidev_cc final epoch jumped to
  0.32 off a 0.14--0.19 tail, while the ten-point window mean was 0.186; across the 08-24..30 sweep
  the window statistic ranged 0.115--0.301, so 0.4 leaves measured headroom and still excludes the
  0.80--1.06 epoch-100 means. The helper rejects fewer than ten values, so shortening a loop cannot
  silently turn the window claim back into a smaller, noisier sample.
- The sibling of that rule, one axis over (gh-ocannl-892): WHICH measurements a tuning run produced
  is as host-dependent as what they measured, so a golden must not pin the list either. A candidate
  whose timing window is mostly host stalls is refused (`Autotune.admitted_timing_ms`), and a
  refused candidate emits no calibration row and no timing — under enough load a whole routine's
  candidates go, and the surviving list gets shorter. `bandwidth_calibration` pinned the four STREAM
  kernels in order, and on cuda and hip a different subset went missing each run while cc and metal
  passed; on idle boxes both GPU backends refuse nothing, and four processes sharing one GPU refuse
  3--15 timings per run on cuda and 18--21 on hip. What survives the load is the RELATIONSHIP
  between the two views of the emission path — a routine contributed rows exactly when its search
  timed a candidate (`report.candidates_timed > 0`) — which is strictly more discriminating than the
  list it replaces: rows going missing while the search timed something, and rows attributed to a
  routine that timed nothing, both fail it, and a contended host moves both sides at once. The
  biconditional alone still passes a routine whose every candidate failed compile or dispatch
  (`candidates_failed`) — nothing timed, nothing contributed — so require the load's own evidence
  for absence: a row-less routine must show `timings_contended > 0`. Put the per-routine
  timed/refused/failed accounting on stderr so a red run names the refused routines instead of only
  showing a shorter list. The same load empties a SEARCH, not only its rows: on the hip iGPU the
  09-02 sweep (and 08-31 before it) ran `test/operations` in parallel and the small searches in
  `autotune_routine_name` and `autotune_serial_baseline` had every window refused, so
  `candidates_timed >= 1`, `default_ms` being `Some`, and "the passed name reached the rows" were all
  false with nothing wrong in the tuner (they pass alone, and six concurrent copies of each passed
  here while logging `NOT TIMED` refusals). Every claim of the form "the search timed / measured X"
  in a tuner test therefore admits `timings_contended > 0` as the one alternative, and the rows
  claims take the biconditional shape above. The admission gate itself is not the bug: refusing to
  rank a stalled window is the gh-855 design, and a search that refused everything ships untuned
  and uncached so a later cold call retries it.
- Applying that rule test-by-test as each sweep day names one is what kept the family red for a
  week (`autotune_serial_baseline` and `bandwidth_calibration` 08-31/09-01, `autotune_routine_name`
  09-02, `autotune_fission_sketch` 09-03). The way to finish it in one pass is a **negative
  control**: set `Autotune.contention_ratio` (a constant in `arrayjit/lib/autotune.ml`, not a
  config key) to `0.`, which refuses every window by construction, and run the tuner tests — what
  fails is exactly what a load-emptied search would fail. Under it `autotune_fission_sketch`'s
  chain search and `autotune_arm_containment`'s run 1 lose their claims about what was timed, and
  `autotune_smoke` additionally loses "second report names its winner", which no sweep day had
  reached yet. Revert the constant afterwards. Read the control as a model, not a verdict: it is
  stronger than any real load, so a test it breaks whose claims a real sweep has never broken is
  not thereby a defect to weaken — the cc-pinned bound-pruning tests (`autotune_bound_pruning`,
  `cost_model_selection`) time on a backend with no round trip to disperse,
  and `bandwidth_calibration`'s remaining claims are about rows existing AT ALL, which needs every
  one of the four stream kernels emptied at once. `autotune_timing_modes` fails the control because
  it pins the contention policy itself, which is the control working.
- Waive on the evidence that explains THAT absence, never on a report-wide count (Codex, staging#608).
  `report.timings_contended` counts WINDOWS and answers exactly one question — was this search's
  measurement set complete — because a refused digest is dropped from `seen` so an equivalent seed
  can retry it: on an idle cuda box `autotune_fission_sketch`'s chain search refuses 4 windows over
  0 distinct candidates, so a claim that added the counter into a candidate population credited four
  candidates that were the same one, four times. Three report fields carry the narrower facts
  (gh-ocannl-892 follow-up): `candidates_contended`, the distinct digests refused that no later seed
  timed — the term that composes with `candidates_timed` and the `Not_dispatched_key` declines into
  "how many distinct candidates did this search reach" — `default_refused`, which separates a
  contention-refused untuned-default reference from the gh-ocannl-552 regression of never proposing
  or attributing that seed, and `baseline_refusal` (gh-ocannl-1423), the baseline's own window and,
  since gh-ocannl-1517, the cause it was refused for.
  `autotune_replay_refusal`'s "the re-search times the baseline the replay released" gates on that
  one, after re-running the re-search up to three times: a `timings_contended` gate would let any
  candidate's refusal excuse it, and a `baseline_ms` gate reads the value under test, so a baseline
  never relinked would skip instead of failing. Its leg 4 refuses the window through
  `on_candidate_measured` and pins the record, the shape for a new gate of this kind. A per-arm or per-report scoping is the same rule one level up: a union
  over two arms lets partial contention in one excuse the other, and `autotune_arm_containment`'s
  injected-failure scenario is waived only by arm B stopping short of the injection threshold
  (`candidates_timed < 2` for `~after_arm_timed:2`), never by arm A's refusals.
- `autotune_fission_sketch`'s chain search has NO margin on a GPU backend by construction: the
  whole-routine presets dedup to the unscheduled base and the beam's moves off it bind no hardware
  dimension (gh-ocannl-543), so exactly one candidate — the fissioned preset — is dispatchable, and
  one refused window takes `candidates_timed` to zero. Its stderr accounting line names the numbers;
  a sweep log that has it is diagnosable, and before 09-03 no tuner test but `serial_baseline`
  printed one.
- The family reaches CI through `@bin-smoke`, not only through the sweep: `projection_shape_bench`'s
  `smoke_2x2` canary is a 2x2 cell, and a shared GitHub runner refused every one of its timing
  windows on 2026-09-03, exiting 1 and reddening a PR whose diff could not have caused it. A refused
  window is now counted apart from a failed cell there — the canary passes `--allow-unmeasured`,
  since it asks whether the pipeline runs end to end, while an ordinary bench run that was asked for
  numbers and got none still exits 1. Both still print the `!!` line. If a `bin/` bench ever reads
  green with `n/a` in the column you asked for, that flag is the first thing to check.
- A sweep log line reading `autotune: partial-report callback failed:
  ("Report_callback_failed(\"Exit\", _, _)")` is NOT a fault to chase: it is
  `autotune_arm_containment`'s designed scenario (a report callback raising the same nullary
  exception the injected arm failure raises, which is why the tuner wraps it) reaching stderr, and
  dune interleaves it with whatever else runs in parallel — in the 09-03 sweep it landed directly
  above an unrelated test's failure and read as its cause.
- An allocation bound is witnessed by a `Gc.allocated_bytes` delta taken tightly around the one
  operation under test, its word count printed to stderr `(not part of the golden)` and only the
  bound claimed on stdout (`broadcast_constant_init`'s `allocation_witness`, gh-ocannl-1218). For
  a host initializer that operation is forcing its lazy (`Ir.Host_inits.find`), done inside
  `Context.compile`'s `~lowered_transform` hook: shape inference has committed the dims and padding
  by then, and linking has not yet forced it — claim `Lazy.is_val` was false beforehand, or the
  delta measures nothing. Size the shapes so the defect the bound excludes lands well above it: an
  allocation per run is visible only when the run count times the smallest per-run allocation, a
  3-word pair, exceeds the bound (512 runs against 512 words). And do not argue from the source
  that a helper is allocation-free: without flambda, a function returning a tuple allocates it on
  every call.
- What boxes per cell in a host loop without flambda, all measured on `Ndarray.set_flat_values`
  (gh-ocannl-1258, witness `ndarray_flat_writer`): a bigarray store whose kind is not known where
  it is written (a polymorphic helper), so match the precision first; any out-of-line call taking
  or returning a float, `int32` or `int64` — Base's checked `of_float`s, an `external` without
  `[@unboxed]`; an `[@inline]` helper whose branches join a boxed result, which boxes the fast path
  too; and a `Complex.t` store, whose record the compiler does not elide (nor a `Complex.t` read's).
  The read side (gh-ocannl-1333, witness `ndarray_flat_reader`) adds one: a float RETURNED from a
  closure the caller does not know is boxed, so a per-precision reader or writer is a closure over
  ints that does the store itself (`Ndarray.flat_reader`/`flat_writer`, `int -> int -> int ->
  unit`), and a callback API that returns a float (`init_array`'s `f`) can be constant only when
  its callback returns a preallocated constant. A read's witness claims its result array (one word
  per cell plus a header) and a bound beyond it.

## Test support and placement

- Find a tensor by name through `Ir.Tnode.label_name`, never `Ir.Tnode.debug_name`. The debug name
  becomes the node's code name once a renderer has minted one. A label that two nodes of one
  routine share gets the `n<id>_<label>` form, and that form sticks. The `.cd`/`.ll` printers
  that `output_debug_files_in_build_directory` turns on name nodes the C emitter never asks
  about (a merge buffer's source). So `batch_norm_running`'s data-parallel lookup of the owner's
  `running_mean_parallel` held with the setting off and raised `Not_found_s` with it on, because
  the shard broadcast `n53_running_mean_parallel =: n13_running_mean_parallel.merge` had renamed
  the owner (gh-ocannl-1552). `label_name` is the identifier the default style mints when no other
  node of the routine shares the label, computed from the label alone, so it keeps the spellings
  tests already used (`w1`, `q.grad`, `kernel_downsample_downsample`). The raw `label` join does
  not: it keeps an initializer's operator label (`*._w1`). A debug name read before any code name
  is minted IS `label_name` (gh-ocannl-1686: `Tnode.get_debug_name` and the heuristic minting share
  one label-to-name function), so only minting moves it. `debug_name_scan` (gh-ocannl-1606,
  in `@test/operations/scans`) holds every `.ml` under `test/`, `arrayjit/test/`, `lib/` and
  `tensor/` to this: a debug name may be printed or raised (a `Session_error` message included), be
  a `Verdict` claim's label, or be the `~substring` needle of a check that some rendered text names
  the node, directly or through `^`, `sprintf`, `String.concat`, a `[%string]` interpolation or a
  `List.map ~f:` of it. Anything else is refused -- an equality, a find, an association key, a
  bound or returned name -- with the closed list in its reader's header. A comparison with text a
  renderer of the same process produced under the display name (a census row) marks the
  application `[@rendered_name "<reason>"]`.
- In the library the same scan keeps a debug name out of every name that outlives the moment it is
  composed: a tensor label (`Parallel.shard_along`'s shards), a block comment of generated code,
  which also seeds routine names (`Train.forward`, `init_params`, the `%cd` `~~(...)` comment), an
  ndarray's `~debug` description, a shape-constraint origin or a taint provenance all take
  `Tnode.label_name`. A message composed when it is shown keeps `debug_name`, which then names the
  node as the code emitted so far does; one the scan cannot follow to its sink (a refusal a
  function returns, a local `logf`) is marked `[@message_only "<reason>"]`, on the application or
  on its `[%string]` (post-wave task post-1-task-6).
- In the compiler (`arrayjit/lib`) a node's `label` and `namespace`, the names spelled from them
  (`debug_name`, `label_name`) and a backend's name are display, never identity: all of them are
  public inputs a user's tensor or a backend can forge. One wave fixed the same defect three times
  (`Online_softmax`'s tile roles by namespace and label head, gh-ocannl-1722; `Schedule`'s masked
  fragment by a `"fragment"` label head, which a user's `{ fragment }` tensor also has,
  gh-ocannl-1721; backend kind by substrings of the backend name, gh-ocannl-1723), each by
  replacing the spelling with a uid membership the compiler fills or with the `Backend_intf`
  capability facts. `label_dispatch_scan` (gh-ocannl-1783, in `@test/operations/scans`) reuses
  `debug_name_scan`'s walker under a second rule and reads fail-closed: a read sits in a display
  position (printed or raised, composed into a message by `^`/`sprintf`/`[%string]`, a `[%log...]`
  payload), is copied onto a node the pass mints (`Tn.create ~label:("tile" :: like.Tn.label)`), is
  a `~backend_name` forwarded under its own label, or is marked at the site -- `[@message_only
  "<reason>"]` for a message the walker cannot follow, `[@label_read "<reason>"]` for a read that
  decides nothing a forged name could steer (`Tnode` spelling its own names and its debug
  registry's key, `Backends.get_backend` selecting the backend the user named, a calibration row's
  backend key). Everything else -- a `match`, an `if`, an equality, a prefix test, a table key, a
  `let` -- is refused. A new decision on a node's role takes a set the pass fills, keyed by uid; one
  on a backend's kind reads `Context.backend_facts`.

- Insert a new stanza in `test/operations/dune` beside its topical neighbours, never at the end of
  the file. The file is grouped by subsystem rather than alphabetized, and the end is where every
  concurrent branch appends, so two unrelated PRs that both append conflict on rebase. The same
  holds for a member line in an aggregate alias (`slow`, `scans`): put it among related members,
  not last, since the last line also carries the list's closing parentheses and every appender
  edits it.

- A mechanism a GPU policy gates gets a cc-runnable structural leg whenever its premise can be
  synthesized, beside the executed GPU legs (gh-ocannl-1275). Per-PR CI runs cc, and on cc a
  GPU-gated premise is `Verdict.skipped`, so every later claim of that section exercises the
  mechanism's absence. Section 6 of `fission_replay_segmentation` replayed an UNFOLDED segmentation
  on cc, and a reintroduced replay policy gate passed there. The synthetic premise is the tuner's
  own pipeline under a GPU target whatever the host (`Ll_test.tuner_segments
  ~target:(Ll_test.synthetic_gpu ())`, gh-ocannl-1483: `Sched.default_gpu`, `Sched.zero_expansion`
  and `~promote_locals:true` at `Ir.Backend_intf.no_hardware_limits`); its zero policy reads the
  configured `gpu_schedule_min_parallel`, so a leg whose premise depends on the threshold pins it
  for the process (section 6 sets it to 1). The leg asserts its premise
  unskipped (here: a `Normal` segment opening with the per-cell zero, carrying a non-empty
  schedule, so the reconstruction claim cannot compare two empty schedules). Then it claims
  structurally: canonical digests and reconstructed schedules (`SC.save_segments` on both sides). It
  needs a negative control that turns it red on cc: the policy-gate mutant there. `qkv_zero_init`'s
  sketch-escape leg (gh-ocannl-1043) is the earlier instance.

- The suite-wide sweep of that rule (gh-ocannl-1626) found five host-side syntheses, each a
  `Ll_test` entry point: the seeded untuned plan derived with `Autotune.seeded_plan
  ~backend:"cuda"` on `Ll_test.captured_lowering` and counted by `Ll_test.seeded_census`; the
  tuner's segmentation under `Ll_test.synthetic_gpu`; GPU seeds at the stand-in tile
  `Ll_test.f32_mma_limits` (`Ll_test.f16_mma_limits` for an f16 site: the f32 tile advertises no
  f16 format, so under it the site's tensorized pipeline is withheld); the scheduled IR of a
  shared-memory schedule read by `Ll_test.guards_and_barriers` instead of a dialect's spelling; and
  workgroup-memory estimates under `Ll_test.gpu_like_capabilities`, the capabilities
  `C_syntax.codegen_capabilities` derives for a shuffle-capable dialect. Each leg is labelled
  `structural` and carries a control in the same leg (the host's backend, a threshold above the
  fixture, no matrix unit, the unfolded pipeline) under which the premise must be absent. What stays
  GPU-only is what needs the device: emitted dialect text, execution, device numerics, timing and
  driver refusals. The renderer's binding-race check, whose thread and storage environment is
  renderer state, is asked through the render itself: `C_syntax.Compile_driver.binding_race`
  renders a routine in a dialect without compiling and answers the refusal's detail, and
  `Ll_test.cuda_like_binding_race` asks it of `Cuda_like_config`'s CUDA syntax on any host
  (gh-ocannl-1780; `hardware_warp_shuffle` judges every racing body and every rendering one). A
  test-local seeding helper that reads the running backend takes an optional
  `Ll_test.tuner_target` defaulting to it, so the same probe feeds the device leg and the stand-in
  one (`schedule_strided_1x1`'s f16 sites). A model decision is judged for a stand-in target
  without compiling it: `Autotune.model_pick` and `model_explain` under `Ll_test.cuda_model_target`
  (scheduling-and-autotune.md). The inventory and each skipped entry's reason are in
  gh-ocannl-1626's closing comment.

- `refused_action_streams` (gh-ocannl-1005) holds accepted-failure Dune actions to the same
  stream contract as fresh-child controls: a run whose owning executable directly declares
  `arrayjit.verdict` and accepts a nonzero exit must capture or drop both streams on its own action
  branch. `Refused_action_scan` derives ownership with `Dune_stanza_scan.program_runners`, preserving
  dependency origin under `chdir`, public names and parent `subdir` blocks; capture on a sibling
  counts for nothing. Literal exit predicates (`:standard`, integers and `or`/`and`/`not`) are
  supported. Includes, directory-spanning ownership, computed library membership, opaque launchers,
  command-resolution overrides, pipelines and unknown refusal actions fail explicitly; the scan
  does not evaluate arbitrary Dune or shell syntax or follow transitive library dependencies.
  `refused_action_scan_cases` holds each boundary beside its nearest legitimate action and runs
  shipping refusal controls through `Fresh_process`, keeping their failure markers out of green logs.

- `Fresh_process` (gh-ocannl-910) owns synchronous fresh-child capture for host-only probes:
  separate temporary stream files, absolute executable resolution, status-plus-causal-text matching
  and exception cleanup. `report` prefixes every echoed child line, so its `STOPPED EARLY`, `FAIL:`
  and `FAILED:` markers cannot be read as the parent's Verdict report by `tools/mutation-run.sh`.
  Deadline, custom-environment and concurrent-process harnesses keep their own contracts.

- `Stderr_capture` (gh-ocannl-1296) owns a test's capture of its OWN stderr (`capture`, `lines`:
  fd 2 dup'ed onto a temporary file, the redirection ppx_expect performs) and the reading of the
  tuner's `autotune-progress:` lines (`Progress`). The capture's contract is the path out: fd 2
  restored, the captured text echoed back, the file removed and the descriptors closed whether the
  function returned or raised, the exception re-raised afterwards with its backtrace -- what the
  hand-rolled copies it replaced mostly lacked, leaving a crash's report on a deleted file.
  `stderr_capture_cases` pins the raising paths, in process and in a child.

- `Gc_witness` (`test/support/gc_witness.mli`) owns what a test reads off the GC. A `Weak`
  witness is taken in an `[@inline never]` builder that returns it and only what the case keeps;
  `reclaimed` retries up to three `settle` rounds (`Gc.full_major` twice, then `Gc.compact`) and
  `kept` requires the value to survive three, so neither rests on one collection. An allocation
  bound is read in `drained_finalisers`, whose leading `Gc.full_major` runs the finalisers of
  earlier garbage outside the window (on metal, tens of thousands of words of `release` messages,
  gh-ocannl-1416). Every user carries the positive control beside its claims: a `kept` witness
  (or `window_sees_allocation`) that must read alive, since a witness observing nothing makes
  every "reclaimed" and every upper bound true. A control that is not already a golden row is a
  silent `Verdict.claim`. Its own library, stdlib only, so an edit reaches the GC-reading tests
  and not every `test_utils` dependant. `gc_witness_scan` holds every `.ml` under `test/` to it:
  a `Gc.full_major`, `Gc.major`, `Gc.compact` or `Weak.create` outside `gc_witness.ml`, qualified,
  through a module alias or under an open, is refused, so a collection a test needs is
  `Gc_witness.settle` (`arrayjit/test/` is another package, outside the corpus).

- `ll_test_ratchet` (gh-ocannl-964) derives test sources from `Source_inventory`, harness membership
  from owning Dune stanza groups (including parent `subdir` blocks and `select` target-to-arm
  relationships), and constructor names from
  `Low_level.t` / `scalar_t`. One inline record construction or one recursive binding matching two distinct IR
  constructors requires
  `ll_test`, public `arrayjit.ll_builders`, or an exact, stale-checked exemption. Constructor names are matched conservatively
  regardless of qualifier; comments, strings and non-record constructors cannot inflate the builder
  census. The six arrayjit consumers adopted the public builders in gh-ocannl-954; their private
  traversals remain local, since builder adoption does not claim traversal centralization. Directory-spanning `include_subdirs` modes and unresolved `include` directives
  are refused explicitly, following the existing Dune scanner boundary. Adoption removes the exemption in the
  same change; counts go to stderr and the
  golden keeps source floors and reasons. `ll_test_scan_cases` drives the shipping scanner as a
  child to pin refusal, adoption, stale exemptions, package coverage and a missing source root.
  Temporary migration rows cap both detected metrics independently: increased records or walkers
  fail even in an exempt file, while decreases remain valid until adoption or removal makes the row
  stale. The canonical harness is explicitly permanent rather than a migration quota. Tuple and
  nullary constructions remain outside this record-literal ratchet's detection boundary.
  Both selected source arms are checked regardless of host availability, against their generated
  target module and actual owning stanza; shared arms require every owner to link the harness.
  Literal `copy_files` / `copy_files#` relationships carry ownership through destination modules
  and copy chains; a linked original cannot cover an unlinked copied consumer. Unsupported copy
  globs/dynamic inputs and inputs outside the declared test corpus are refused explicitly. Documented
  `test/ppx/*_expected.ml` goldens are not implementation modules and are excluded; arbitrary unowned
  sources remain checked. The `scans` aggregate runs the shipping scanner and its control suite.
  Adoption is linking the harness AND calling its IR surface (gh-ocannl-1052): a link made for an
  operand helper (`Ll_test.cycle`) had retired rows whose hand-built IR never moved, which is the
  general trap of a ratchet whose adoption test is a proxy — the proxy spreads for other reasons. The
  surface is DERIVED, per harness module, from `ll_builders.ml`/`ll_test.ml`: a value whose
  definition mentions an `Ir`-rooted path (or an alias of one, its own or included), or calls what
  resolves to an IR value, is IR; each binding is classified on its own, a later definition
  replaces an included one's class. The rest (`cycle`, `weighted`, `blank`, ...) is printed in the
  golden, so a new helper's class shows in review. Both halves resolve names through one lexical
  scope model (`Lexical_scope.scoped`, extracted from this scan for `codegen_text_scan` in
  gh-ocannl-1079): every value binding form scopes its names (externals and
  instance variables included), module aliases (constrained ones too) are shadowed by later
  bindings, functor parameters and unpacks and die with their structure, and only an exact harness
  path is the harness. Its one deliberate gap: an `open` of a module the scan cannot read is not taken to shadow
  a harness open before it, since the tree writes `open Ll_test` then `open Verdict.Claims`. A linked
  source that calls nothing of it gets its own refusal and still needs a migration row.

- Pure IR node, index, statement and scalar builders live in public `arrayjit.ll_builders`
  (gh-ocannl-954), re-exported unchanged by `Ll_test`. Every Dune consumer spells the public name,
  so package-filtered arrayjit builds with tests can resolve it. Optimization, execution,
  discriminating test data and traversals stay in `ll_test`; the builder tier depends only on
  `base` and `arrayjit.ir`. `set` accepts an optional debug label to preserve diagnostic fixtures.

- Pure assembly dialect and anchor controls live in `test/operations/asm_census_cases.ml`
  (`@test/operations/runtest-asm_census_cases`), independent of the `cc_march_census` toolchain
  matrix. Residual histogram controls there pin counts and bounded stderr formatting, without
  compiler-sensitive mnemonic goldens (gh-ocannl-924). The driver retains its cache-format probes. `Asm_census.census_source_in` shares the
  exact-anchor-before-brace-range policy between the driver and the discriminating parser control
  (gh-ocannl-937).
- A GPU branch census reads the code object the run compiled, never a re-compile (gh-ocannl-1695):
  with `output_debug_files_in_build_directory` the CUDA and HIP compiles carry line tables
  (`-lineinfo`, `-gline-tables-only`; `Compiler_options`' `?with_line_info`), and
  `Asm_census.disassemble_gpu` reads `<routine>.ptx` through the toolkit's `ptxas` for the
  EXECUTING device's `sm_XX` and `nvdisasm -g`, or `<routine>.hsaco` through `llvm-objdump -d -l`
  (minix: only the distribution's `/usr/lib/llvm-<N>/bin`). Read the architecture from the
  backend's `Context.static_properties`, never `nvidia-smi`, which ignores `CUDA_VISIBLE_DEVICES`;
  NVRTC's PTX may target an older one, which the driver JITs for the device. A split compile
  writes `<routine>.part<k>.*`, whose line tables name the part's own source. Dialect fixtures are
  in `asm_census_cases`'s `gpu_probes`.

- `flip_bound_pruning` needs a finite arm-A incumbent before its deliberately tiny envelope can
  fathom materialize flips (gh-ocannl-962). An all-refused arm leaves infinity, so a finite roofline
  cannot prune it; macOS CI run 34269878888 attempt 1 exhibited the failing count and attempt 2
  passed at the same head. The test requires a
  twofold floor/incumbent margin before deciding, otherwise reports environment-scoped undecided;
  missing timings must have arm-A refusal evidence, while floor validity and numeric parity remain
  unconditional. `autotune_fission_sketch` counts completed refused windows as timed, but those
  windows cannot supply a best single for recombination. `fiss_sketch_composite` distinguishes
  ineligibility, missing singles with per-segment refusal evidence, a proposal, and its own refused
  or admitted window. The CPU count is singles plus one exactly when eligible, and eligibility
  still requires the composite to reach timing; unrelated refusals waive nothing. Post-admission
  injection preserves the partial report's admitted outcome and count; an untriggered injection
  skips only on that coarse composite's explicit refusal outcome.

- The bounded gh-ocannl-973 provenance contract supports functor-valued module arguments through
  `module_bindings` and first-class standard ordering through ordinary deferred-call closures.
  Three conservative refusals remain named in `verdict_ratchet_controls.md`: an empty recursive
  local standard namespace after an unknown open; partial ordering before an optional constant
  default resolves; and a supplied replacement for an unapplied Boolean callable default.
  Those refusals are boundaries, not evidence that the corresponding OCaml claims are invalid.
  General module typing and polymorphic structural comparison remain outside this reader.
- Hand-run shell tests source `scripts/harness-support.sh`: `harness_args`,
  `harness_scratch`, `report`/`skip`, and `finish` share the counted SKIP contract,
  complete header help and signal cleanup. `cleanup_fixture` releases owned children even with
  `--keep`. To iterate on one leg, `--only REGEX` runs the legs whose
  `if harness_leg 'NAME'; then ... fi` gate matches the extended regular expression and prints
  `FILTERED  NAME` for every other one; its footer is a `FILTERED RUN, not a full pass` line,
  never `all legs passed`, and a filter that selects no leg (or a harness that declares none)
  exits 2. It is an option only, with no environment form, so an ambient filter cannot shrink a
  CI or `tools/test-harnesses.sh` run. A gate holds a leg's body only: fixtures and helpers a
  later leg reads stay outside it, or a filtered run dies on an unbound variable.
  `tools/test-test-run.sh` gates every leg by its number (`--only '^81:'`).
  `shell_scripts_parse` discovers `tools/test-*.sh` and `scripts/test-*.sh`
  (except the production supervisor) and refuses private copies or missing lifecycle calls.
  Standalone harnesses elsewhere declare `# ocannl-harness: standalone`, as
  `test/operations/ci_matrix.sh` does; Dune actions use the function-only API without that marker.
  `mutant` retains checked AWK edits; `expect_rejected` requires a diagnostic pattern or reason callback,
  while `harness_rejected` requires both status and reason for in-process twins, including the
  Dune-run sweep's Metal fingerprint check without nested Dune. Owned-child deadlines and
  identity-checked cleanup are shared too; callers must probe their independent state reader
  and retain unreaped direct-child ownership until `harness_wait_child` returns.

- Broad autotuner integration searches live on `@slow`: `autotune_batched_companion`,
  `autotune_fission_sketch`, `autotune_mma_companion`, `autotune_arm_containment` and
  `tune_ship_arm`. Their sources and golden assertions remain intact; each runs through
  `@test/operations/slow-<name>`, with `no-infer` keeping its execution off `@default` too.
  The regular suite retains `autotune_smoke`, `inline_flip_tune`, `autotune_candidate_release`
  and the focused scheduling/cache tests. Run the affected slow integrations before pushing
  changes to their search, placement, fission or failure-containment paths.

### Process-lifetime settings

- A configuration key read inside a `lazy` that a module creates at initialization is read ONCE
  per process (gh-ocannl-1535): the first force resolves it, and that value -- or the exception a
  refused value raised (`Utils.Invalid_config_value`, a `User_error`), which OCaml re-raises on
  every later force -- answers for the rest of the process. A test that states such a key in
  process, by `Hashtbl.set Utils.config_file_args`, therefore changes nothing once anything has
  forced the lazy, and an earlier test case, a compile that reached the key at its default, or a
  library call made while building the fixture all count. The failure is silent: the run goes on
  under the first value. The keys are listed below, per source file, and `test_config_consistency`
  holds the list to the code in both directions.
- State a process-lifetime key from outside the process. For one value per executable, give the
  test its own configuration: a directory with an `ocannl_config` of its own, as
  `test/operations/profiles` has (its rules then list `ocannl_ambient` too, as
  `config_dep_completeness` requires), or a `setenv` around the rule's `run`. For several values in one test, run a child per value: `Fresh_process.run`
  re-executes the test's own executable with `--ocannl_<key>=<value>`, which outranks every other
  source, and the child reports what it saw on stdout -- `cache_key_agreement` states
  `cc_grid_private_bytes_cap`, `cc_pool_core_class` and `max_shape_error_origins` that way. An
  in-process statement is sound only as the first thing the executable does, before anything can
  force the lazy, and the site says so (`placement_surface` states `gpu_fission_keep_mapping` that
  way, gh-ocannl-1430). There is no reset hook, by decision: add one beside a lazy only when a test
  needs two values of it in one process and a child cannot serve.
- Values computed EAGERLY at initialization are a different class, outside the list: they are read
  when the module is linked in, before any test code runs, so no in-process statement reaches
  them at all -- the bootstrap keys and the profile pick, the debug runtime's settings, and the
  snapshots behind `Utils.settings` (`Utils.restore_settings`), `Low_level.virtualize_settings` and
  `Tensor.default_value_prec`. Most of them land in a mutable handle a test sets directly instead.
- What counts as a process-lifetime read is `Config_key_scan.process_lifetime_reads`' closed list
  of forms, over the libraries' sources (`arrayjit/lib`, `tensor`, `lib`): a `lazy e` or
  `Lazy.from_fun e` no `fun` encloses (a nested module's or a functor body's included), and the
  keys `top_level_key_reach` credits its body in the scope it sees -- every literal `~arg_name`
  read, lambdas included, and the reach of every value it names, in its own source or, through
  `library_reach`, in another library module (a `Utils.settings` field standing for every key its
  initializer reads). It errs long on names (one only mentioned is credited), so a cc probe lists
  the probe cache's whole key, and `schedulers.ml` lists what `Cc_backend`'s pool policy reads. It
  does not see a memo built from a `ref` or a table, or an eager read. Each entry below is one
  bullet, `- <source>.ml: <key>, <key>.` with every name in a code span; any other bullet in this
  section is prose.
- `affine.ml`: `legality_crosscheck`.
- `backends.ml`: `debug_check_upload_arenas`.
- `autotune.ml`: `autotune_bound_pruning`, `autotune_calibration_file`, `autotune_log`,
  `autotune_log_seeding`, `autotune_progress`, `model_cache_bandwidth`, `model_cache_capacity`,
  `model_default_geometry_lattice`, `model_default_schedule`, `model_device_gather_rate`,
  `model_dram_bandwidth`, `model_fill_threads`, `model_issue_rate`, `model_mma_rate`,
  `model_multiprocessors`, `model_peak_flops`, `model_peak_memory_bandwidth`,
  `model_transaction_rate`, `model_workgroup_bytes_per_multiprocessor`.
- `c_syntax.ml`: `cc_grid_private_bytes_cap`, `schedule_log_declines`.
- `context.ml`: `debug_fill_fresh_outputs`.
- `cc_backend.ml`: `cc_backend_arch_flags`, `cc_backend_compiler_command`,
  `cc_backend_probe_cache`, `cc_backend_simd_flags`, `cc_kernel_cache_mib`, `cc_parallel_grid`,
  `cc_pool_core_class`.
- `cuda_backend.ml`: `cuda_kernel_cache_mib`, `cuda_printf_fifo_size`, `debug_log_from_routines`,
  `log_level`.
- `hip_backend.ml`: `collapse_foreign_stderr_noise`, `debug_log_from_routines`,
  `hip_kernel_cache_mib`, `hip_printf_fifo_size`, `hip_scratch_validation`, `log_level`.
- `metal_backend.ml`: `metal_kernel_cache_mib`.
- `persistence.ml`: `checkpoint_load_mmap`.
- `row.ml`: `debug_check_retired_solver_vars`, `max_shape_error_origins`,
  `shape_substitution_cross_check`.
- `schedule.ml`: `automatic_cpu_schedule`, `automatic_gpu_schedule`, `gpu_fission_keep_mapping`,
  `schedule_fission`, `schedule_log_launches`.
- `schedule_outcome.ml`: `strict_failure_classification`.
- `schedulers.ml`: `cc_backend_arch_flags`, `cc_backend_compiler_command`,
  `cc_backend_probe_cache`, `cc_backend_simd_flags`, `cc_parallel_grid`, `cc_pool_core_class`,
  `multidev_num_devices`.
- `sketch_families.ml`: `cc_parallel_chunks`, `cc_parallel_grid`.
- `tensor.ml`: `debug_check_forward_root_owners`, `debug_gc_backprop_roots`, `op_shape_schemes`.
- `tnode.ml`: `default_prec`, read eagerly and wrapped by a lazy.
- `utils.ml`: `collapse_foreign_stderr_noise`.
