# Build and test mechanics

The dune/OCaml mechanics behind AGENTS.md's workflow rules, and what CI actually covers.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

AGENTS.md holds the workflow rules; these are the dune/OCaml mechanics behind them, narrow enough
that they earn a lookup rather than always-loaded space.

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
  index row the day it appears.
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
  is the outcome that gets a check disabled rather than fixed. Merging does NOT repeat the
  exercise: under the roll-forward policy (gh-ocannl-861) the gate is one green full-matrix run
  for the PR's current head, and a clean merge proceeds on it however far the base has moved —
  re-verifying after every sibling merge is exactly the cost the policy removed (staging#533 ran
  three clean rebases and three full CI cycles over an unchanged topic diff before it). The gate
  reads whatever the head is: a merge that adds no commit to the branch restarts nothing, while
  any commit that moves the head — a conflict resolution, a rebase, a merge of the base — waits
  for its own green run, conflicts or not. A diff the `ci` path filter ignores entirely
  (`docs/**`) gets no run at all; there an absent check is the filter's answer, not a missing
  verdict. Bring the base in again before merging only when its advance touched the PR's own
  files — which the endpoint diff `<staging>/master..HEAD` cannot tell you, since it includes the
  PR's edits and so makes every nonempty PR look drifted. Anchor the question at the branch point
  instead, as the intersection of two name lists — `git diff --name-only --no-renames $(git
  merge-base <staging>/master HEAD) <staging>/master | grep -Fxf <(git diff --name-only
  --no-renames <staging>/master...HEAD)`, with `<staging>` the remote name resolved above — which
  prints exactly the PR's files that the base's advance also touched, and nothing when there are
  none. The pieces are there for reasons: `--no-renames` lists both names of a file the PR
  renamed, so an edit the base made to the old name still shows, and `grep -Fx` matches whole
  lines, so a path with spaces is compared as one path instead of being word-split into
  pathspec fragments. A scan that turns red on the merged tree anyway is a master red like any
  other, owned by the CI-red triage routine (the CI section below) rather than by the session
  that merged.
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
- Repository scans derive source membership through `Test_utils.Source_inventory` (gh-ocannl-871):
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
  Current one-word config assignments such as `profile=reproducible|performance|approximate` and `backend=cc`
  are file/key/count-pinned judgments because their spelling alone cannot distinguish them from an
  arbitrary API or mathematical assignment.
  Whitespace around `=`, within the value, or an empty example value does not hide the key. The
  permanent negative controls are Metal's `fastMathEnabled=false`, Apple's `mathMode=Safe`, and the
  mathematical `d=1`; `debug_log_from_routines=true` pins a real documented positive. An outer bare
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
  rather than as a golden line, so a scan that goes blind cannot be promoted back to green. And
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
  And keep the rule's DECISION somewhere synthetic text can reach it. `env_var_deps`' XOR lived in
  its main loop, so "a rule running its test through `bash` is subject to the rule" could be argued
  and not asserted; `Scan.backend_rule_of` is that decision alone, with the diagnostics and tallies
  left in the check that owns their wording, and `dune_scan_cases` states the rule over stanzas the
  repository does not contain.
  gh-ocannl-723 is the pattern's second instance and shows what a scanner rule costs once the seam
  exists: the rule is `Scan.artifact_subjects`, `dune_scan_cases` puts it to fifteen stanza/source
  pairs the repository has no member of, and `config_scan_lexing` does the same for the source side
  — where the hostile input is the repository's own, since `test/support/generated.ml` names
  `Generated.init` in half a dozen doc comments and `generated_provenance.ml` asserts on a string
  literal quoting one, so a text scan would read the module that DEFINES the initializer as its
  heaviest caller. Two lessons beyond the pattern. First, a decision-table control still leaves the
  wiring from decision to failure unexercised, and `env_var_deps --control` closes it: it builds a
  synthetic tree containing the violating pair, hands it to `env_var_deps.exe` as a CHILD, and
  claims that the child names the stanza and exits 1 without the declaration and exits 0 with it —
  the `generated_provenance` capture pattern, and worth the fixture because the tree is DERIVED from
  the check's own exemption and gateless lists and so cannot drift from them. Second, a vacuity
  floor over a repository census cannot be asked of an arbitrary tree: the artifact caller floor
  applies only when `Config_key_scan.floor_violations` says the run was handed the repository's scan
  roots, and WHICH mode the run was in goes into the golden, so a glob that breaks flips that line
  rather than quietly retiring the floor.
  gh-ocannl-800 adds the corresponding refusal ratchet. `env_var_deps` derives scanner sources from
  the repository-wide rules through the shared `per_directory` traversal, extracts literal formats
  handed to `Verdict.fail`, `failwith`, or a Verdict claim form (including `Printf` formats, with
  substitutions removed), and requires a unique full-format marker plus its readable fragment in the scanner's
  assigned permanent control golden. `Refusal_control_manifest` is the explicit bridge, but not its
  own evidence: a claim marker is emitted only after `Verdict` recorded a passing execution of that
  exact format (and each observation is consumed once), while a direct-failure marker requires the
  exact successful negative-control line assigned to that refusal or an explicit observation from
  the caught branch itself. Marker occurrences are consumed as a multiset, so two identical
  formats require two exercised controls. A new manifest row therefore
  prints nothing until its arm supplies runtime evidence. `refusal_control_scan_cases.expected`
  holds the manifest equal to mechanical extraction, every entry present in the assigned live/case
  golden union, and the manifest's repo-relative source paths equal to `env_var_deps`' derived
  scanner census. The
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
  Two more, about the rule rather than the scan. A declaration is justified by ANY read of the key it
  tracks, so phrase the rule over the key (`Config_key_scan.source_reads_key`) and not over the one
  function that prompted the check — otherwise the documented way of pinning a key becomes unusable
  for that key. And having widened it that far, widen it in BOTH directions: permitting a
  declaration for a direct reader while requiring one only of the function's callers leaves exactly
  the stale run the check exists to prevent. A third: `inline_tests` on a library does not make its
  modules test-only, so a rule that is about what a module DOES when linked (here, emptying the
  artifact directory) must not accept a declaration that invalidates the inline-test runner alone.
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
- That floor machinery is `Test_utils.Scan_floors`, shared by the configuration scanners and by
  `codegen_text_inventory`: pass it the root table, and get the census-by-root, the itemised
  diagnostic and the stderr report. Each scan keeps its own table, since which roots it globs and
  how far each may fall are facts about that scan.
- PIN THE RELATIONSHIP, NOT THE RESTATEMENT: where a check needs a set that some other part of the
  system owns, relate the two rather than writing the set down again and asserting that the copy
  still says what it says (gh-ocannl-706, after gh-ocannl-591 and gh-ocannl-689 turned out to be the
  same defect twice). Three shapes carry it: a list or a `match` with one entry per member of
  something else's vocabulary; a golden that IS the restatement, printing a list the test itself
  owns; and a count over a derived set. What makes it worse than an unchecked list is the direction
  the drift pushes — gh-ocannl-689's marker vocabulary was closed deliberately, so a stale copy
  rejected the CORRECT marker for a newly added backend and the author's cheapest remedy was `none`,
  a lie the grammar accepts. Prefer DERIVING over pinning where the derivation is free: 29 test
  files copied `Ir.Schedule.backend_is_gpu`'s substring test instead of calling it, and the fix was
  to call it. Where the list must stay put, four properties make the pin real. Assert it from
  wherever the link cost is ALREADY paid, so no new dependency follows the check around —
  `marker_backend_vocabulary` is a whole executable existing to be the one place that links the
  backend closure. Compare as SORTED LISTS of identities, never as counts or as sets, so a duplicate
  on either side is a mismatch too. Claim one bare boolean through `Verdict`, with both lists on
  stderr, so the golden stays fixed as the underlying set changes and only the literal ever needs
  editing. And give it a control on a SYNTHESIZED violation rather than on the corpus, since a
  corpus where every member satisfies the rule cannot tell a rule from a tautology. The two landed
  exemplars to copy are `test/operations/marker_backend_vocabulary.ml` (gh-ocannl-689, the marker
  vocabulary against `Backends.all_of_backend`; gh-ocannl-706 added the CPU/GPU classification
  beside it) and gh-ocannl-723's `artifact_subjects`, which asks the question from the dune scanner
  that already pairs a stanza with the sources its `(modules …)` name.
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
- `dead_export_scan` (gh-ocannl-806) is a source-level ratchet over direct `.ml` modules without a
  sibling `.mli` in `arrayjit/lib/` and `tensor/`: it enumerates source-declared top-level `let` and
  `external` values plus the values generated by top-level `of_sexp`, `sexp`, `compare`, and
  `equal` derivings, then counts qualified references from other sources. Source-declared `let`
  names starting with `_` are excluded by policy (gh-ocannl-941), matching OCaml warning 32;
  this reaches patterns and extension payloads, but leaves external and deriving-name census
  policies intact, including polymorphic-variant parser helpers. `[%of_sexp: M.ty]`,
  `[%compare: M.ty]`, and `[%equal: M.ty]` count as references to the corresponding generated
  value; the walk also reaches type constructors nested inside the extension's type and those
  consumed by another deriving. `sexp.opaque`, `sexp.ignore`, `compare.ignore`, and `equal.ignore`
  suppress the corresponding implicit reference, and an inherited polymorphic-variant row calls the inherited type's `__<type>_of_sexp__` parser helper.
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
  when shadowing could make it local, and `include M` counts every value because it re-exports the
  interface. Those choices admit false positives rather than refusing valid code. Values created
  by other PPX expansions or brought into the defining module by `include` are outside this first
  cut. Every current zero-reference export is an exact stale-checked exemption: adding an `.mli`,
  removing the value, or giving it a detected caller requires deleting its exemption.
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
  caps dune and records the verdict (gh-ocannl-606 collapsed the former three-party launch and its
  publication handshake) — and everything it keeps per worktree (the lock, its owner pointer, the
  `last` pointer) lives under `~/.ocannl-test-runs` keyed by the worktree's path, never in the
  tree: a run leaves the worktree exactly as clean as it found it, so a teardown that judges a
  worktree by its ignored files is not refused over lock residue. `OCANNL_TOOL_TEST_RUNS` relocates
  the lock with the diagnostics, so two sessions on one worktree see each other's lock only under
  the same override. Prefer foreground `run` launched through the agent harness's background mode
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
  — the log's `exit:` sentinel, `status`, `list` — stays dune's 1: it is what happened); `TIMEOUT`
  (142, the cap's SIGALRM exit and the only code allowed to say timeout); `KILLED` (137);
  `CANCELLED` (129/130/143); `ERROR (toolchain/setup: nothing ran)` (126/127). `wait`'s own
  deadline is 124, and `status` exits 0 published / 3 running / 1 died without a verdict — it
  reports publication, never the verdict. `tools/test-test-run.sh` legs 29–32 pin all of this
  against a fixture dune that emits dune's exact refusal stderr, with the red-run and guard controls.
  `repeat [--alone] N` is the supported flake diagnostic: it holds that same worktree lock once,
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
- Every liveness question in that script — per pid and per process GROUP — reads process STATE and
  not only the signal, because `kill -0` succeeds on a ZOMBIE exactly as on a live process, and an
  identity token does not rescue the check either: a zombie leader still prints its recorded
  `lstart`. Answered with the signal alone, `stop` could announce `orphaned process group N ignored
  TERM; escalated to KILL` for a group holding nothing but corpses — the one report someone consults
  when working out why a worktree lock will not clear (gh-ocannl-742). `group_alive` is ONE shared
  definition, `scripts/process-group.sh`, sourced by both `tools/test-run.sh` and
  `scripts/setup-ocaml-env.sh` for the identical misreading (staging#621 replaced the two
  synchronized copies): `/proc` where there is
  one (fork-free, through the shell's `read`), `ps -A -o pgid=,stat=` on the BSDs and macOS,
  degrading to the bare signal only where neither answers; the signal probe stays FIRST as a
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
  timeouts) but since staging#621 (gh-ocannl-795) the Ubuntu 5.5 `main` CI leg runs BOTH in one
  isolated step before `setup-ocaml`, summing their exits so a red first harness does not suppress
  the second's diagnostics. The step is now the home of any hand-run harness that has a reason not
  to be a dune test — `tools/test-ci-times.sh` joined it for a different one (its subject invokes
  `python3` under that exact name) — and they share one contract for a leg the host cannot decide:
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
  worktree key EXTRACTED from the shipping script rather than guessed. Against it: a group whose
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
- **Promote through `tools/promote.sh` during a merge, on every platform.** Promotion writes the
  WORKING TREE; `git commit` during a merge takes the INDEX. So a golden promoted after its `git
  add` is committed with its PRE-promotion content, and nothing local objects — every later `dune
  runtest` reads the working tree and passes — while CI builds the committed tree and fails on the
  golden diff. It cost ~90 minutes on staging#487, and it is invisible by construction: the
  usual conflict drill (resolve, `git add`, run the suite, promote what moved) puts the promotion
  on the wrong side of the `add` every time, and `git status` shows the file staged, which is what
  you were checking for. `promote.sh` closes it: mid-merge it takes `dune promotion list` BEFORE
  applying (afterwards nothing pending is left to name), then stages exactly what it promoted and
  says so. A promoted golden still UNMERGED is deliberately NOT staged and is warned about with the
  `git add` that would accept it — staging one records a conflict resolution, which is the caller's
  call and not the script's. Outside a merge nothing runs, not even the extra `dune` invocation.
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
- Formatting is a gate on the PR path, and its mechanics are worth knowing before the first red.
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
  status still wins unchanged. `tools/test-fmt-check.sh` pins the clean, warning, and formatter-error
  outcomes separately. Two files a formatter cannot handle are refused at the site rather than
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
- Two Windows C-runtime formatting differences make hand-formatted floats non-portable in goldens:
  it prints 3-digit exponents (`e+018` where Linux prints `e+18`), and it rounds representable
  decimal ties away from zero where glibc rounds to even (`%.1f` of `2.25` prints `2.3` there,
  `2.2` on Linux). `Ir.Ndarray.concise_float ~prec` normalizes the exponents, OCaml's `%h`
  hex-float format sidesteps decimal rounding entirely, and tie-free test data sidesteps it too;
  `test/support/test_utils.ml` packages the rules — `hex_float` and `set_binary_stdout` are
  portable by construction, while `print_float`/`print_floats` delegate to `concise_float` and so
  still need tie-free inputs.
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
  the same rule, and `Test_utils.Scan_argv.expand` splices it back in where the `@` argument stood.
  A new whole-tree scan should instead use `Source_inventory`; `env_var_deps` also knows the older
  response-file shape — a target
  the rule hands back to its own action needs no golden diff — and recognizes it by the `@<target>`
  reference rather than by the name, so a real output cannot be renamed out of the requirement.
  The one thing this transport gives up is a path containing WHITESPACE: `%{deps}` is space-joined
  and dune has no boundary-preserving expansion, where argv preserved boundaries. It is given up
  loudly — every word of a response file is a dependency dune materialized, so `Scan_argv.expand`
  refuses a word that names nothing on disk rather than scanning two phantoms in place of one file.
  The repository has no such path today.
- A Windows checkout with `core.symlinks=false` (git's default there without Developer Mode) writes
  every git symlink as a small TEXT FILE holding the link target. The repository has 19 of them —
  `docs/in-progress/` and `docs/research/` are views onto `docs/proposals/` — so any scan that reads
  documentation sees 29-byte stubs, and `config_usage_scan` reports an exemption count drifting to
  zero. It is not a repository bug and must not be "fixed" in the exemption list: CI's Windows job
  enables symlink evaluation (setup-ocaml runs `fsutil behavior set symlinkEvaluation`) and sees the
  real files. On a hand-run Windows box, `git config core.symlinks true`, then delete and re-check-out
  the 19 paths (`git ls-files -s | awk '$1=="120000"'`) before trusting any doc-reading scan.
- `os.kill(pid, 0)` is not a liveness probe on Windows and must not be used as one. CPython
  implements `os.kill` there as `OpenProcess` followed by `TerminateProcess(handle, sig)`, so signal
  0 KILLS the process it was asked about, and a pid that already exited raises `OSError`
  (`WinError 87`) rather than `ProcessLookupError`, which no POSIX-shaped handler catches.
  `benchmarks/cell_group.process_is_alive` is the portable form — a zero-timeout wait on a process
  handle, where `WAIT_TIMEOUT` means "still running". `signal.SIGKILL` does not exist there either;
  `os.kill` with any other signal is `TerminateProcess`.
- `(copy_files ...)` creates PASSIVE rules: they do not fire just because you build a sibling target
  in the same directory — only when listed in that target's `(deps ...)` or requested explicitly. A
  rule consuming copy_files output must therefore declare it. And validate a `(mode promote)` target
  from a clean state (`dune clean && dune build @alias`): stale `_build/` intermediates can satisfy
  an undeclared dep, so an incomplete build passes while the artifact is wrong. Assert content
  (size, object counts), not mere existence.
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
  process-private, since dune runs one process per executable. Any configured `build_files_prefix`
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
  The dune side of that: `init` READS `build_files_prefix`, so the stanza dune runs the test under
  must declare `(env_var OCANNL_BUILD_FILES_PREFIX)` — otherwise dune serves the previous run's
  result when the variable changes, which is gh-ocannl-628's hole one key over. `env_var_deps`
  requires it of every stanza whose `(modules …)` name a source that calls the initializer, and
  reports a declaration with no caller behind it as well (gh-ocannl-723). Where the declaration goes
  is dune's semantics and not one rule: a `(test)` runs under its own `(deps …)`, an inline-test
  library under `(inline_tests (deps …))`, and an `(executable)` has no `deps` field at all, so it is
  the rule that RUNS it that carries the declaration — the same placement as the `ocannl_config` dep
  and the backend marker, and checked as such (a declaration on a NEIGHBOUR of that rule reruns the
  neighbour, so it does not count). One name of an `(executables (names a b) …)` is one program, and
  attribution follows dune's main-module rule: `a` is built from module `a`, a module that is no
  name's main module is linked into all of them, and `(public_names …)` pairs positionally — so
  `b.exe`'s rule answers for `b` alone, and only a shared module puts the requirement on every runner
  (gh-ocannl-747; combining them reported `a` undeclared over a rule linking neither its main module
  nor its initializer).
- **An ambient-environment GUARD needs its keys declared, or it never runs.** A test that refuses to
  run when an OCANNL variable that would rewrite its golden is set — `startup_streams`,
  `profile_precedence`, `config_profiles` — reads those keys through `Utils.read_env_var`, the one
  reader no commandline flag or config file can outrank. But the guard only executes when dune reruns
  the rule, and dune reruns only for a variable the rule DECLARES: a key on the guard's list with no
  `(env_var OCANNL_<KEY>)` beside it is a key the guard never sees, and dune serves the previous
  golden across a change of it. `env_var_deps` pairs the two (gh-ocannl-749) rather than trusting the
  hand-written list. `Config_key_scan.env_reader_reads_in_source` RESOLVES each reach or REFUSES it:
  a string literal at the call names its key, and a key taken from a list names the elements of that
  list, resolved through the shapes the guards here are written in — a top-level `let` of string
  literals, `a @ b`, `List.map keys ~f:fst` over a table of pairs, iterated by a `List` combinator
  the scan knows. Anything it cannot follow is reported per reach, not approximated: an earlier
  version fell back on the source's string literals, which is a superset where the list is in the
  file and says nothing where it is not, so one incidental literal made an unresolved reach look
  answered. Every construct it follows is named, every name it trusts (`List`, `fst`, `snd`, `@`,
  the standard roots) is checked for rebinding, and a file that rebinds one gets no resolution at
  all. Keys are normalized before the registry is consulted and are asked for KNOWN OR NOT — the
  reader builds `OCANNL_<KEY>` whatever the registry says — so a key OCANNL does not read must be
  pinned rather than declared, the sibling check refusing a declaration that names none.
  A variable a run pins with `(setenv …)` is exempt where the pin SCOPES over that run, and pinning
  is the better option wherever it is available; every rule that runs the program must answer, since
  dune invalidates each on its own deps. A `(library)` is refused outright, inline tests included:
  it is linkable by executables that declare nothing, which is the argument `Artifact_in_library`
  makes for the initializer. The negative control is a third synthetic tree in
  `env_var_deps --control`, permanent rather than transient, since every guard in the tree declares
  and a corpus-drawn control would record the absence of the shape.
  What the resolver is FOR is worth knowing before extending it: catching a guard whose declarations
  drifted from its key list, which it does exactly. It is deliberately not adversary-proof — a source
  can always put its keys behind an abstraction — and the module header says so. If that trade stops
  holding, the answer is a structural contract for how a guard spells its keys, matched rather than
  inferred, not another name in its tables.
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
  report it. A non-empty collection prints exactly what `p` prints, which is what let ~44 files
  convert with their goldens unmoved; an empty one prints `<claim> (empty): false`, and `?min:n`
  prints `<claim> (only 1 of 4): false`. Arrays go through `Array.to_list` rather than growing a
  second family. What stays on the unguarded spelling is the claim whose passing reading IS
  emptiness — "no candidate declines", "no key is undocumented", a scan over a tree that is usually
  clean — and those want a companion claim that the population was there at all, which is what the
  `p_empty ~over` form is.
  `verdict_ratchet` enforces the guard mechanically, for a quantifier (`for_all`/`for_alli`,
  `for_all2_exn`, `is_empty`, a negated `exists`/`existsi`, in Base's or `Stdlib`'s spelling)
  written directly into a
  native claim (`Verdict.p`, `pf`, `claim`, `claimf`, `pass_fail`, opened or qualified) and for one
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
- Mutation manifest rows use `tools/mutation-run.sh <module> <patch-file> <@alias>`
  (gh-ocannl-969), in an otherwise idle, isolated worktree. The patch is literal bytes
  `OLD@@@NEW`, with exactly one delimiter, a nonempty OLD occurring exactly once (overlapping
  occurrences count), and no implicit newline trimming; NEW can be empty. The runner invokes
  `tools/test-run.sh`, prints its actual verdict and run id, and extracts every complete
  `FAIL: ...: false` line from that run's full log, preserving order and duplicates. Exit status
  remains test-run's, so a killed mutant normally exits 1; a build failure with no false claims
  is not evidence for a manifest row. A passing mutation still exits 0 and needs investigation.
  INT, TERM and HUP cancel and reap the run before restoring; restoration is confirmed by `cmp`
  against the backup, including CRLF and a missing final newline. `tools/test-run.sh idle`
  probes its existing worktree flock (0 idle, 3 held, 2 unreadable); this is a point-in-time
  snapshot, not a reservation. A busy preflight refuses before mutation. If a killed launcher
  or supervisor leaves a lock holder alive, restoration is deferred: inspect/stop the worktree
  runs before recovering the source manually. Refusal exits 2; deferred or failed restoration exits 3 and retains the printed recovery copy; after SIGKILL, use that copy
  manually. It is temporary storage, without a power-loss or reboot recovery guarantee. `tools/test-mutation-run.sh` drives the shipping
  runner and test-run supervisor in isolated fixtures; Ubuntu's shell-harness CI runs it.
- The guarded pairwise claim has the same two label dialects as the scalar claim (gh-ocannl-816):
  `pf_all2` formats a computed label before taking the two arrays, and `pass_fail_all2` preserves
  `pass_fail`'s lazy failure detail while adding the structural empty, floor-shortfall, or
  length-mismatch reason. A compound claim that joins independent parity pairs stays out of a
  many-pair combinator: split it into separately labeled `p_all2` claims, so the transcript names
  which readback failed rather than merely saying that one of them did.
- Guarantees that fire only on an empty collection are never exercised by a green suite, so
  `verdict_quantified` stages them: the satisfied forms run directly, and each refusal runs as a
  CHILD process whose streams the parent captures. Capturing is not tidiness — a refusal prints
  `FAIL:`/`FAILED:`, this repository's failure marker, so an inherited stream puts those words in a
  green run's log and costs `grep FAIL` its meaning (the same argument `generated_provenance`
  makes). The second half of that test is the one that makes a wide sweep safe: it runs `p` and
  `p_all` in two children and requires their stdout to be equal, which is the property "converting
  a site does not move its golden" stated as a check rather than as a hope.
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
- The dynamic-indexing pair has builders of its own: `Ll_test.gather` (`Get_dynamic`),
  `Ll_test.scatter` (`Set_dynamic`) and `Ll_test.scatter_add`, the read-modify-write shape
  `rewrite_one_hot_reductions` actually mints. Reach for them rather than spelling the record:
  the `idcs` array must carry a `Fixed_idx 0` PLACEHOLDER at `dyn_axis` — a convention stated
  only in `low_level.ml`'s type declaration, and one a hand-spelled site gets to state again for
  every axis it happens to have — so the builders plant it and refuse a `dyn_axis` outside the
  array. Neither constructor reaches `optimize` through the ordinary pipeline (lowering emits
  neither, and the rewrite runs after both virtualization arms), so hand-built IR is the only way
  to put one in front of the analyses; `gather_table_placement.ml` is what that looks like.
- An executed reference must DISCRIMINATE, not merely run: give every producer a value that varies
  with every symbol of its iteration and stays clear of the init/sentinel value (`1 + i`,
  `1 + 10*outer + inner` — the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`). A
  constant producer replays an identical assignment under a too-wide range guard, a value omitting
  one symbol is constant along that axis under a wrong substitution, and a value colliding with the
  zero-init hides a dropped first iteration — each is a leg that passes for the wrong reason.
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
  `~output_dims` then `~input_dims`), not the `Array.init` argument. What this does not buy is
  aperiodicity: the values repeat with period `modulus`, so a shift by `modulus` is a symmetry, and
  where the blocking factors are searchable a packed panel can repeat under `k -> k + p` and hide a
  panel-substitution bug just as well; the recipe with no shift symmetry at any lag is
  `bin/narrow_gebp_bench.ml`'s `mix`. A formula that is NOT a flat index —
  `(i0 + i1 + 2*i2 + 3*i3) mod 7` — is not this class and needs no conversion, as long as no
  coefficient is a multiple of the modulus.
- Dune roots at the OUTERMOST ancestor holding a `dune-workspace` (failing that, a `dune-project`)
  and ignores dot-directories, so from a worktree under `.claude/worktrees/` the main checkout wins
  and the worktree is invisible to dune: targeted commands fail with `Don't know about directory
  .claude/worktrees/...`, while a bare `dune build`/`dune runtest` quietly builds and tests the
  PARENT branch. `scripts/setup-ocaml-env.sh` writes a one-line `dune-workspace` at the worktree
  root, restoring it as the root with its own `_build`. The step tests the ancestor DIRECTORIES
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
  and CI runs it only in the pre-toolchain Ubuntu step alongside `tools/test-test-run.sh`
  (the `group_alive` bullet above has the SKIP contract the two share).
  It copies the WORKING-TREE hook into throwaway clones under a `mktemp -d` (never touching this
  repository's refs or config) and covers the watchdog (TERM at the bound, KILL 5s later, the
  process GROUP, rc preservation, no orphans), the counting wording and its two recovery commands,
  the offline `skip` with the count taken as of the last successful fetch, a branch and a tag both
  named `origin/master` not displacing the tracking ref, `FETCH_HEAD` left byte-identical, and
  which SSH launcher git ends up invoking with or without the appended OpenSSH options. Both
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
  that serializes on `ocannl_training_test` must take the lock, since an unlocked action in a
  locked file is what the next training test gets copied from (the gate starts no pool). Hand-written
  per-test aliases depend on the gate explicitly, while dune's GENERATED `runtest-<name>` aliases do
  not — a targeted run of a `(test)` stanza can be served stale under a rejected spelling — and
  `env_var_deps` checks that every alias with test actions in a directory carries that directory's
  gate.
- Deleting a file target out from under dune is not a way to force it to re-run: `dune build
  <that target>` afterwards exits 0 having produced nothing (observed on dune 3.23.1 with
  `test/operations/<name>.exe.output`), and `-f/--force` does not rescue it — `--force` only
  re-runs actions attached to ALIASES. Either force the alias (`dune build --force
  @<dir>/runtest`), or run the built exe directly with its cwd set to `_build/default/<dir>`, which
  is exactly the environment dune gives it — the same cwd, hence the same `ocannl_config` search
  root, that makes `dune exec` unusable (AGENTS.md). The cause is that dune trusts its own digest
  database and never stats a rule's targets, so a hand-deleted one is recorded as built forever;
  that also rules out the two other reflexes, since touching a source changes no CONTENT digest and
  deleting `_build/.digest-db` does not restore the memo either. Every golden-diff rule now has an
  alias to force (`dune build --force @<dir>/runtest-<name>`, see below); for a target with no alias
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
  render. Types rather than sources, because `C_syntax.compile_proc` has neither an `.mli` nor a
  return annotation: its document exists only as an inferred type, and a source scan would have to
  be told about it — which is the hand-maintained frontier again, the one three of the four review
  rounds on gh-ocannl-712 found a member of. Add a renderer to a library and it is on the frontier
  the day it is exported; nothing to update.
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
  `module_alias_targets` resolves an `open` of an aliased emitter module — including one a FUNCTOR
  produced (`module Syntax = Ir.C_syntax.C_syntax (…)`, which is how every backend and every codegen
  test reaches `compile_proc`), by taking the functor's own name — which the rejection below would
  otherwise miss. Membership, taint, the buffer destinations and the pin walk all go through
  the one resolver: rules that know different routes are how a file stays listed while its pin
  disappears.
- **What no file-local rule can follow now says so.** A buffer is where generated text lands with no
  name to carry it, and the ways to fill one do not end (a wrapper reaching its parameter through a
  local binding, PPrint's own `ToBuffer` renderers, a buffer in a record). So a substring test whose
  haystack reads a `Buffer.contents` this scan never saw filled — directly, or through the bindings
  the read travels along, by the same fixed point taint uses — marks that file's itemisation
  **partial** rather than dropping the fragment silently — the file is listed, the fragment is
  unnamed, and the inventory says which.
- A test that classifies COMPILER-PLAN text rather than generated text marks that one binding with
  `[@@ocannl.codegen_text.compiler_plan]` (gh-ocannl-865). The inventory excludes text tests only
  inside that binding: the annotation is not file-wide, so a real generated-source pin elsewhere in
  the same test remains visible. `codegen_text_scan_cases` controls both directions — an unannotated
  compiler-plan classifier is still reported as partial, and an annotated classifier beside a real
  generated-text assertion still reports the real fragment.
- **An `open` (or `include`) that hides a route is refused, not approximated — over its own scope,
  and never over a name the file binds.** A structure open governs the items after it, `let open M
  in` its body, and a nested structure's opens die with it. A name the file binds anywhere is struck
  from every refusal: `open Ir.Low_level` followed by a local `let to_doc` is valid code calling the
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
  the re-run is still called for. Markers are read only outside `Verdict` claim lines, since a
  claim label is prose ABOUT a kernel and freely quotes its vocabulary ("padded GPU intrinsics fire
  against the threadgroup fragment").
- A golden tagged `[derived] beside <source>` got in by neither route. The markers describe whole
  dumps, and a golden can hold emitted text in fragments instead — a table whose columns are the
  `%cd` and C-style spellings of one constant, a census of the schedule decisions a kernel was
  built from. Such a file is a member when the test beside it demonstrably reaches generated text
  AND its golden holds something other than that test's own verdicts. That last condition is what
  keeps the rule useful rather than noisy: a boolean column does not move when codegen does, so a
  schedule test's all-`true` golden stays out while its source stays in.
- Every rule that diffs a golden — the repo-wide scans, the codegen snapshots, the config-precedence
  rules, the ppx-output diffs — carries its own `runtest-<name>` alias (gh-ocannl-726), so
  `dune build @test/operations/runtest-verdict_ratchet` runs that one test, applies its diff and is
  promotable, and a misspelling exits 1 (`Alias … is empty`) rather than doing nothing. The seven
  repo-wide scans additionally share `dune build @test/operations/scans` (gh-ocannl-703): the family
  runs in seconds, and is what to run after touching a config key, a dune stanza, a printed claim or
  an agent note. Two properties of dune shape how those aliases are written, and both are silent
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
  point, so the directory needs a `runtest-env_spelling_gate` rule for it to depend on and an
  `(alias (name runtest) (deps (alias runtest-<name>)))` stanza, both of which `env_var_deps`
  checks.
- Splitting a golden per backend is a decision about WHAT the golden holds, not a formatting choice,
  and the bar is high in both directions. Its cost is a golden only some machine re-records — cc,
  multidev_cc and metal on the reference Mac, cuda and hip only on the sweep's GPU boxes — so a
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
  target puts the slow run on plain `dune build`'s `@all`.
- Two focused aggregates sit beside `scans`, built the same way and answering a narrower question
  (gh-ocannl-783): `dune build @metal-codegen` runs the Metal-pinned tests — the executed Metal-only
  guards and the emitted-MSL structural ones — and `dune build @lifecycle` runs the
  resource-lifecycle probes, the tests that drive `Ir.Resource_fault_injection` or read
  `Ir.Alloc_census`. Each family alias is spelled identically in `test/operations/dune` and
  `arrayjit/test/dune`, which is what makes the root-level `@<family>` run both halves: `dune build
  @foo` builds the alias in the current directory and every directory beneath it. Both run in
  seconds against warm executables, which is the point — the backend-wide directory run that was
  previously the only way to reach these legs is neither.
  - Membership is DERIVED, not written down twice: `env_var_deps` calls a stanza a Metal member when
    its `; ocannl-backend:` marker names `metal`, and a lifecycle member when its modules name the
    instrumentation, and fails on a member the family stanza omits — naming the `runtest-<name>` to
    list and the `@<family>` that would otherwise have skipped it. The derivation is a FLOOR, so a
    family may list more: `arrayjit/test`'s `test_slab_free_on_grow` is a lifecycle member by intent
    and names no instrumentation module, since it drives `Backend_impl.Make_slab` against a mock raw
    backend of its own.
  - Writing a new family means giving it the same treatment. The derivation has to be a property the
    member stanza declares for an INDEPENDENT reason (a backend marker, a module it uses, a glob it
    writes) — a marker comment invented to say "I am in family F" is the second copy of the list
    again, and a check reading it can only confirm that the copy still says what it says. A family
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
  loop when the elements have effects); OCaml 5's `Lazy.is_val` returns TRUE for a lazy that is
  mid-force (forcing it then raises `Lazy.Undefined`), so it cannot guard a force reachable from
  inside that lazy's own computation; two record types sharing a field name resolve to the
  LAST-defined type, silently mistyping `x.a.b` (which is why the scheduler's event field is
  `dev_state`); and `Base.Float.max_value` is INFINITY — the finite maximum is `max_finite_value`.

### What CI actually covers

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
  codegen and its scheduling are still reached by no CI leg.
- A red on merged master is presumptively CLAIMED work. `ci.yml`'s `notify-triage-routine` job
  fires the "ocannl-staging CI-red triage" Claude Code cloud routine on any non-PR master red —
  push and scheduled sweeps alike (a logged no-op until the `ROUTINE_FIRE_URL`/`ROUTINE_FIRE_TOKEN`
  repo secrets are set; a lost fire is caught by the routine's own daily backstop sweep). The
  routine claims the red with an issue on **ahrefs/ocannl** titled
  `CI red on master@<short-sha>: <workflow>` — issues are disabled on the staging repository, and
  ahrefs/ocannl is where the issues live anyway (AGENTS.md's two-repository rule) — and either
  opens a `ci-fix/*` PR on staging (it never merges its own PRs) or posts its diagnosis to that
  issue.
  Merging sessions do not watch CI after landing (roll-forward, ahrefs/ocannl#861); before fixing
  a master red by hand, find the claiming issue and any linked PR, and take over only where triage
  visibly stopped short — saying so on the issue first.
- The Ubuntu/OCaml 5.5 main job preprocesses `cc_backend.ml` with
  `OCANNL_LOG_LEVEL_CC_BACKEND=3` and builds both `@check` and
  `@test/operations/runtest-cc_backend_trace_name` in that environment. The focused runtime test
  executes one cc routine and checks the bare result inside its `work` trace against the compiled
  routine name: `@check` alone only type-checks, so a trace expression that silently resolved a
  different in-scope binding would otherwise remain green (gh-ocannl-859). The test reports an
  explicit skip in ordinary level-0 suite runs, keeping the extra execution confined to this gate.
- `cuda_backend.ml` and `hip_backend.ml` are compiled by NEITHER the macOS dev boxes nor CI, so a
  green `dune build @check` locally proves nothing about them. Each lives in an `(optional)` library
  over `cudajit`/`hipjit`, and `arrayjit.context` reaches its implementation through a dune `select`
  whose fallback arm is `<backend>_backend_impl.missing.ml`; with the vendor package absent both
  mechanisms succeed SILENTLY, so an exit status cannot distinguish "compiled" from "skipped".
  Verify on the box carrying the toolchain — `rog-nv-wsl` for cudajit, `minix-amd-wsl` for hipjit —
  and check two things there rather than one: that
  `_build/default/arrayjit/lib/.<backend>_backend.objs/byte/<backend>_backend.cmi` exists, and that
  the `select` landed on the vendor arm, which
  `head -1 _build/default/arrayjit/lib/<backend>_backend_impl.ml` names outright — dune copies the
  chosen arm under a `# 1 "…"` line directive naming it, and `.missing.ml` there is the stub. Each
  box carries exactly one of the two toolchains, so the OTHER backend's absent `.objs` on that same
  build is a free negative control. `@check` also proves compilation and never execution, so pair it
  with a runnable probe wherever one exists. Two PRs in two days paid for this: gh-ocannl-758
  (staging#490) shipped a HIP arm unparsed beyond syntax and edited the CUDA arm blind the next day, and
  gh-ocannl-773 (staging#494) touched both again. gh-ocannl-794 is the executable follow-up for CI
  coverage, gh-ocannl-796 for scripting the off-box loop.
- `tools/remote-verify.sh` is the one-off counterpart to the scheduled sweep for a pushed branch:
  it derives the remote pointing to the staging repository by URL, fetches the named branch,
  without rewriting the checkout's `FETCH_HEAD`, resolves one commit, creates a fresh detached
  worktree, resolves the checkout's selected opam switch before leaving it, runs explicitly under
  that switch, and removes just that worktree before its exit sentinel (never repository-wide
  `worktree prune`, which could unregister an unrelated temporarily unavailable worktree). The individual
  commands (including worktree add/remove) and the whole SSH trip have separate process-group caps; the latter also bounds setup
  and cleanup. Non-login shells receive the CUDA/WSL PATH prefix that `tools/sweep.sh` uses.
  Ambient `OCANNL_*` variable names are printed and cleared before opam runs; names injected by the
  selected switch are printed and stripped inside `opam exec`, so only the requested backend can
  override the pushed tree's configuration. A regular, non-symlink root `ocannl_config` from the pushed commit
  is the configuration boundary; when the commit has none, the script creates an empty ignored one
  in the disposable worktree so a personal file above it cannot reach root-launched probes. A
  worktree root nested under any outer Dune root is refused: without that
  boundary Dune can build the parent checkout while this script reports the detached commit.
  `--expect-lib cudajit|hipjit` asserts all three
  pieces of optional-backend provenance above (positive `.cmi`, vendor `select` arm, and the other
  backend's absent `.cmi`). A test, probe or `--record-golden` trip also requires a backend and
  asserts `_build/default/test/config/ocannl_backend.txt`; an unrestricted test alias is reported
  only as passing under that configuration, since the alias may be backend-independent, while a
  runnable probe must print its own backend/device evidence. An `@check`-only trip says explicitly
  that it compiled code and executed no backend. Golden mode prints the corrected `.actual`
  contents and an apply-ready patch, then re-runs the alias before accepting it so a second failing
  dependency cannot hide behind a promotable diff. Before reset, source status (with untracked-file
  reporting forced independently of Git configuration) must name exactly the listed golden
  destinations, both after promotion and after the re-run. After reporting it, the script resets tracked
  files, removes untracked files, and proves the worktree clean at the resolved commit before
  running another operation. Every path also reasserts exact HEAD, clean tracked/untracked source,
  and the unchanged configuration boundary after each operation and before the final certificate;
  a nominally successful probe that edits its checkout therefore fails loudly. Do not replace its
  unpiped ssh output with a convenience pipe: the verifier source travels on a separate remote file
  descriptor while child stdin is `/dev/null`, and the far-side sentinel is the build verdict plus
  cleanup, and the local sentinel is ssh's transport verdict.
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
  `-isystem`; today `cc_march_census` is the test that consumes that hook. The real
  download is deliberately x86_64 Linux/Debian-family-only, since its cc kernels execute on the
  host; `--dry-run` validates and prints the complete staging plan on macOS and other hosts. This is compiler/codegen evidence, not an OS emulator: the
  GCC patch release is whichever candidate the configured apt indexes serve (the exact version and
  target are printed and major 13 is enforced), and the clang leg has a Linux cross sysroot rather
  than the macOS SDK, ABI, linker or runtime. It therefore complements CI and gh-ocannl-794 rather
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
- The per-PR suite does not run the training integrations. `mlp_names`, `mlp_bn_names`,
  `circles_conv`, `fsm_transformer` and `transformer_names` sit on the `train` alias — a third
  tier beside `runtest` and `slow`, for runs that are toy-sized by intent but serialized on the
  `ocannl_training_test` lock, which made them CI's wall-clock tail on every substrate measured
  (2026-08: ~95s/195s/246s cc for the last three on a 24-thread Linux box; the ubuntu runner's
  whole `dune runtest` step was ~8min, most of it this chain; taking it off the per-PR path took
  ubuntu CI from ~31min to ~15 and macOS from ~16 to ~10). The daily sweep's full-suite units
  build `@runtest @train` together, so every backend still executes them daily, and per-PR CI
  runs the tier as a macOS-ONLY shard: measured with the compile cached, `@train` is ~4.7min on
  an M4 Max but ~10min on a 24-thread Linux box, so a ubuntu shard (slower still on the 4-vCPU
  runner) would replace the ~15min main job as the matrix's latency ceiling while the macOS shard
  fits under it. A change to training dynamics, `Train.*` plumbing or the autotuner's fission
  path is still worth a local `dune build @train` (or the affected
  `@test/training/train-<name>`) before pushing, since ubuntu-specific breakage there surfaces
  only in the sweep. `@slow` keeps the
  genuinely long real-dataset runs (`mnist_conv`, `cifar_conv`, `gpt2_dry_run`). CI also runs its
  one dune walk as `dune build "@default" "@runtest"` — two separate commands would hold the
  serialized lock chain (what remains of it) against an otherwise idle runner after every file
  target finished, and the quotes are for PowerShell on the Windows leg, which splats unquoted
  `@` tokens to nothing.
- Both `ci.yml` and `gh-pages-api.yml` cache the built local dependency switch `_opam`, where the
  ~180 compiled packages live; setup-ocaml separately caches opam's root and bare compiler switch.
  Their entries stay separate because the CI matrix and the fixed docs runner have different key
  shapes, but both keys include the platform, compiler, project opam files, ordinary-package
  solution digest and resolved-pin digest from `.github/actions/pin-revisions`. That action runs
  after setup-ocaml updates the repositories and after the workflows' `opam pin -n` steps. It asks
  opam's solver for the selected package/version set under the install's test/doc flags, hashes the
  normalized definitions of those exact versions too — minus the project's own packages — then
  derives every remote git pin from opam's live registry, resolves it with `git ls-remote`, and
  digests the shas. The project packages are left out of the definition hash because `opam pin .`
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
  holds nothing else), the per-definition digest listing (asserted to carry a DISTINCT hash per
  definition, so a per-run constant naming nothing would go red), color suppression, empty
  registries and failed resolutions all have fault-injected negative controls. CI runs the harness
  once on its Ubuntu main leg because this
  is POSIX action plumbing, not an OCaml test or a repository scan, and the fixtures need neither
  setup-ocaml nor network.
  Both workflows use exact-key restores only: cache
  lookup happens after derivation, so a prefix fallback could overwrite the live registry with an
  older switch. And both install non-Windows depexts unconditionally after restore, because those
  are system packages absent from `_opam` (gh-ocannl-809).
- Every workflow that installs dependencies sets opam's global `archive-mirrors` to
  `https://opam.ocaml.org/cache` right after setup-ocaml, before any pin or install. setup-ocaml
  points opam at the GIT opam-repository, whose `repo` file declares no mirror (the HTTP one at
  opam.ocaml.org does), so a bare CI switch downloads every archive from its package's upstream
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
- `gh-pages-api.yml` runs for pull requests and `workflow_dispatch` as well as pushes to master, but
  only a push deploys; validation runs receive their own concurrency group and exercise the build
  and cache topology without publishing. Publishing pushes share `gh-pages-deploy` with the other
  Pages workflow, so the two deploys cannot race over the same branch (gh-ocannl-808,
  gh-ocannl-825).
- Windows is off the per-PR matrix (62-74min against 20 on macOS and 29 on ubuntu) and runs on a
  twice-weekly schedule, together with an ubuntu job on the OCaml floor the opam files claim
  (`>= 5.3.0`, against 5.5 everywhere else). Both are reachable on demand through
  `workflow_dispatch` with `extended: true` — dispatch from a branch when touching `.expected`
  goldens or the cc backend's toolchain handling rather than waiting for the sweep, since those are
  the changes that actually break on Windows (line endings, float formatting, mingw). Twice weekly
  rather than weekly because actions/cache evicts entries unread for 7 days, and an exactly-weekly
  cadence would pay the cold-switch cost every time. The two ride the same cadence because they
  fail the same way: slowly, and through the dependency cone or the toolchain rather than through
  a change under review.
- The Windows job ends with a smoke of `tools/test-run.sh` itself, from Git Bash: `run`, `status
  last`, a deliberately failing target, `start`/`wait last`, `list`, each asserted against its
  documented exit code. That script's MSYS branches (unconditional and fatal `opam-env.sh` sourcing,
  dune routed through `dune-quiet.sh`, the tokenless liveness degradation where MSYS `ps` has no
  lstart/state columns, the flock through MSYS perl) execute nowhere else, and are otherwise reached
  only from rare hand-run Windows sessions — where their rot is discovered mid-task. It builds
  `tools/promote.sh`, a source file dune copies rather than compiles, so it costs a workspace scan;
  the failing-target leg is the load-bearing one, since it is where a dropped `PIPESTATUS` in
  `dune-quiet.sh` would report red runs as green. It runs even when `dune runtest` above went red
  (Windows runs twice a week; a golden drift must not mask the runner's health for three days) and
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
- `tools/sweep.sh` is the coverage for every backend CI does not run: cc, multidev_cc and metal
  locally, cuda on `rog-nv-wsl`, hip on `minix-amd-wsl`, all pinned to ONE resolved commit so a
  mid-sweep merge cannot leave the machines testing different trees. multidev_cc needs no hardware
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
  shell text): it forks, `setpgrp`s the child, and signals the GROUP on expiry, TERM then KILL, so
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
  `tools/sweep.sh --slow --force` (raise `OCANNL_TOOL_SWEEP_CAP` where a backend cannot fit the
  default 90 minutes).
  When the execution column was introduced, existing `pass` rows became `legacy-pass` with
  `execution=unknown`; old incremental evidence is retained, but cannot masquerade as a forced run.
- **The hip unit runs its tests under `dune -j 2`, and a red minix/hip whose failures are all
  `HIP_ERROR_INVALID_DEVICE` at `hip_init`, `HIP_ERROR_NO_BINARY_FOR_GPU` at module load, or
  failed stream creation is the WSL2 dxg bridge overflowing, not the backend.** minix's GPU is
  an iGPU reached through `/dev/dxg`: every allocation and module load is a synchronous message
  over a Hyper-V VM bus ring, and the ring fills when the suite's test executables hold the
  device at once (dune's default there is 32 jobs). The kernel says so —
  `dmesg | grep 'misc dxg'` shows `vmbus_sendpacket failed: fffffff5` (-EAGAIN) bursts exactly
  spanning the unit — and the HIP runtime reports the lost messages under those three names,
  while `rocminfo`, a standalone `hipGetDeviceCount`/hiprtc/`hipModuleLoadData` probe, and the
  single tests all pass. The 2026-09-05 sweep recorded it twice (60+ red tests, zero test-logic
  failures) after the box had kept its VM across two host resumes (dmesg shows
  `hv_utils: TimeSync IC version` renegotiations between the last green unit and the first
  burst): a fresh VM had tolerated the full width for eleven daily sweeps, so the degraded
  bridge lowers the tolerance rather than removing it, and a fresh VM removes the refusal
  class. Under WSL2 `/dev/kfd` and `/dev/dri` are never present and `rocm-smi` always reports
  the driver as uninitialized — neither is evidence of a lost passthrough; `hipGetDeviceCount`
  is. The `unit_jobs` table in `tools/sweep.sh` holds the cap (`OCANNL_TOOL_SWEEP_JOBS=<n>`
  overrides it for one run), applied to the test phase only: `test_cmd` compiles under `@check`
  at full width first, since the cap bounds GPU-holding processes, not the build. The value is
  measured, not guessed: on the degraded bridge dune's default lost 67 stanzas (356 kernel-side
  refusals), `-j 4` still lost 27 (120), and `-j 2` ran a forced full `@runtest @train` unit
  clean in 18.5 minutes (the sweep harness pins the call shape). A single GPU serialises the
  kernels anyway, so the capped test phase is not much slower. Recovery for the bridge itself
  is `wsl --shutdown` from the Windows side, then a kick from the coordinator
  (`wake-lab.sh kick-wsl minix`) — from inside the VM the shutdown kills the session issuing
  it. A retry scoped to module loading would cover one of the three names: the refusal lands on
  `hipInit` and on stream creation too, per VM-bus message, so it would have to sit at the
  binding's error check, and it cannot help a process whose `hip_init` was refused
  (gh-ocannl-927).
- **Runtime-refusal signature table.** These are the exception names `tools/sweep.sh`'s
  `ENVIRONMENT_REFUSALS` treats as the environment refusing a run rather than a test judging it;
  dune prints an uncaught binding error as `Fatal error: exception <name>:` with the status on
  the next line, and the sweep keys on the name. A `fail` unit whose log carries any of them is
  *environment-red* and gets a **serial rerun** (gh-ocannl-945): every failing stanza again as
  `dune build -j 1 @<dir>/runtest-<name>` (or `@<dir>/<alias>` for an explicit rule), one dune
  call each, under the same worktree lock and the unit's own cap, appended to the unit's log.

| name (`Fatal error: exception <name>:`) | call site | statuses seen (minix, 2026-09-05) |
| --- | --- | --- |
| `hip_init` | `Hip.init`, backend `ensure_initialized` | `HIP_ERROR_INVALID_DEVICE` (11), `HIP_ERROR_NO_DEVICE` (1) |
| `hip_module_load_data_ex` | `Hip.Module.load_data_ex`, backend `link` | `HIP_ERROR_NO_BINARY_FOR_GPU` (34) |
| `hip_stream_create_with_priority` | `Hip.Stream.create`, backend `get_device` | `HIP_ERROR_OUT_OF_MEMORY` (4) |
| `cu_init` | `Cu.init` | analogue by construction — rog-nv reaches its GPU through the same dxg bridge; not yet observed |
| `cu_module_load_data_ex` | `Cu.Module.load_data_ex` | analogue, not yet observed |
| `cu_stream_create_with_priority` | `Cu.Stream.create` | analogue, not yet observed |

- **Reading a rerun's verdict.** Adding a name means adding it in both places: the table in
  `sweep.sh` gates the rerun, this one records what the name has been seen with. The verdict is written as `serial
  rerun:` lines in the log AND the fingerprint (outside the fingerprint's 60-entry bound, so a
  wide red cannot drop it), and quoted in the sweep's summary: `still red: <aliases>` names the
  stanzas red on their own — including one generated alias per member of a `(tests (names …))`
  stanza — read those first, they are the test-logic failures the noise was hiding; `all clean`
  says every red target passed alone; `unjudged (exit N): <aliases>` are the ones the cap cut
  short, never folded into `all clean`. One or two inline-expectation sites use a capped
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
- A forced full-suite sweep also intersects the backend-scoped `Verdict.skipped`
  executable-and-claim keys from every successful unit through `tools/aggregate-skips.sh`
  (gh-ocannl-792), writing
  `logs/<stamp>-skip-coverage.txt`. Incremental logs are refused because a cached Dune action does
  not replay its stderr, and failed or interrupted units are refused because they may not have
  reached every test. An intersection over only the completed backends is a loud `POTENTIAL` report;
  it becomes `FAIL` only when every backend in the sweep's own unit vocabulary completed, while the
  sweep itself still exits zero so later units and their history rows are never suppressed. A
  `--ref` predating the machine record is refused when its human skip lines have no paired records.
  A skip caused by a host or configuration capability rather than the selected backend (a compiler
  target, preprocessing flag or filesystem feature) uses
  ``Verdict.skipped ~aggregation:`Environment``: its human stderr line stays visible, while the
  record's scope keeps it out of the backend intersection. The sweep's stdout summary quotes the
  report's `result:` verdict and each `FAIL:`/`POTENTIAL:` finding line (indented, under the
  `skip coverage:` line that carries the report path), so consumers of sweep output — the daily
  scheduled routine's report and notification foremost — see zero-coverage findings without
  opening the report file; the routine diffs the latest report's finding set against the previous
  `*-skip-coverage.txt` and treats `FAIL`, or a changed finding set, as notify-worthy.
  The same report intersects environment-scoped records across the successful unit logs and judges
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
  completed declared-box log becomes `FAIL` only when every declared box contributed; with missing
  boxes it is `POTENTIAL`, and a pre-declaration historical ref is left explicitly unaggregated. A
  claim may be backend-gated in one run and environment-gated in another (the default-policy
  `autotune_mma_companion` leg is the exemplar). An environment record in any declared-box log
  assigns that executable-and-claim key to the environment dimension; backend or environment skip
  records for the same key then both mean their box did not execute it. This ownership-before-
  intersection order prevents a different scope from masquerading as execution. A current unit
  absent from a historical target's declaration contributes backend evidence and is ignored for
  that target's environment matrix; in the other direction, a declared box with no runnable unit
  is a harness refusal rather than permanent silent non-coverage.
  A configuration matrix that runs outside the fleet sweep uses
  ``Verdict.skipped ~aggregation:`Outside_sweep``: the announcement and machine-record validation
  remain, but neither sweep intersection claims ownership. This is deliberately narrow: the
  `cc_backend_trace_name` claim is executed by the Ubuntu compiler-trace CI leg at trace level 3,
  while the ordinary sweep's default configuration cannot execute it (gh-ocannl-885).
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
- In an errexit shell test, `! cmd` is not an assertion. Bash exempts a command whose value is being
  inverted, so `! grep -q 'must not appear' "$out"` runs, returns 1 and the script carries on — the
  negative half of a test can be entirely inert while reading as covered. Spell it as a function
  whose body uses `if`, so the command errexit weighs is the CALL and the ERR trap names its line,
  and have it print what matched: `$BASH_COMMAND` from inside a function names the body, not the
  pattern. Same shape as the `p_all`/`p_none` rule for `Verdict` claims — a check that cannot fail is
  worse than a missing one, because the golden and the roster both count it.
- An unreachable machine records `skip (unreachable)`, and a sweep of skips is not a failure. It is
  not the expected steady state either: both GPU boxes are cabled and Wake-on-LAN armed, and wake
  over Ethernet from sleep and from full shutdown alike, so a run that is meant to cover CUDA or HIP
  wakes them first rather than waiting for a day someone left them on. Waking one is not the same as
  reaching it: the sweep addresses the `-wsl` aliases, and WSL starts on demand or at login, never
  at boot, so a woken box answers on its Windows alias while `rog-nv-wsl`/`minix-amd-wsl` are still
  refused. Kick it with `ssh <box>-win 'wsl.exe -d Ubuntu -e true'` and re-probe for a minute or two
  while tailscaled registers. On a box woken from power-down (as opposed to resuming from
  sleep/hibernate with the owner's interactive WSL shell still open, in which case the VM survives
  the resume and none of this applies), a kicked VM can also terminate again within minutes if
  nothing connects to it — `-win` up, `-wsl` refused again — so re-kick right before actually using
  `-wsl` rather than trusting a wake from earlier; once a real ssh session is running inside, the
  VM stays up for the duration. What IS a failure is silent non-coverage: track the age of the last
  `pass` per backend, because nothing else in the project tests CUDA or HIP at all — and read a long
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
  previous run. Standing identical reds stay quiet, which is the signal-to-noise property the
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
  `flip_bound_pruning`, `cost_model_selection`) time on a backend with no round trip to disperse,
  and `bandwidth_calibration`'s remaining claims are about rows existing AT ALL, which needs every
  one of the four stream kernels emptied at once. `autotune_timing_modes` fails the control because
  it pins the contention policy itself, which is the control working.
- Waive on the evidence that explains THAT absence, never on a report-wide count (Codex, staging#608).
  `report.timings_contended` counts WINDOWS and answers exactly one question — was this search's
  measurement set complete — because a refused digest is dropped from `seen` so an equivalent seed
  can retry it: on an idle cuda box `autotune_fission_sketch`'s chain search refuses 4 windows over
  0 distinct candidates, so a claim that added the counter into a candidate population credited four
  candidates that were the same one, four times. Two report fields carry the narrower facts
  (gh-ocannl-892 follow-up): `candidates_contended`, the distinct digests refused that no later seed
  timed — the term that composes with `candidates_timed` and the `Not_dispatched_key` declines into
  "how many distinct candidates did this search reach" — and `default_refused`, which separates a
  contention-refused untuned-default reference from the gh-ocannl-552 regression of never proposing
  or attributing that seed. A per-arm or per-report scoping is the same rule one level up: a union
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

- Pure IR node, index, statement and scalar builders live in public `arrayjit.ll_builders`
  (gh-ocannl-954), re-exported unchanged by `Ll_test`. Every Dune consumer spells the public name,
  so package-filtered arrayjit builds with tests can resolve it. Optimization, execution,
  discriminating test data and traversals stay in `ll_test`; the builder tier depends only on
  `base` and `arrayjit.ir`. `set` accepts an optional debug label to preserve diagnostic fixtures.
