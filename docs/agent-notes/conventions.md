# Conventions

Releases, configuration spellings, git and PR mechanics, and the honesty rules for test output
and measurement reports.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Releases use lightweight, un-prefixed git tags (`0.8`, not `v0.8`).
- The changelog is written in editorial passes, never in feature PRs (gh-ocannl-807): at release
  prep, or an explicitly-requested batch catch-up, derive `CHANGES.md` entries from the records the
  work already left — merge commits (`git log --first-parent`), PR bodies, issue closing comments.
  Bullets are user-facing (what changed for someone using the library; internal test/tooling
  plumbing usually earns none), one to three lines each, citing `gh-ocannl-NNN` or a
  `lukstafi/ocannl-staging` `PR #NNN`; the mechanism, rationale and measured numbers stay in the
  PR, the issue and these notes. Those last two rules are decidable from the text and are checked
  by `test/operations/changelog_unreleased_scan`, over the `## [Unreleased]` section alone —
  released sections are history and stay untouched. That section is also held to the shape it has
  always had — blank lines, `### ` subheadings, `- ` bullets and two-space continuations — and any
  other Markdown in it fails the scan by name rather than being parsed: the scan decides one
  grammar, so its imprecision cannot pass a bullet nobody checked.
- API stability is not promised and deprecation cycles are not run: exported surface that is dead,
  superseded or in the way is REMOVED, in whatever release finds it. The README's Development
  section states this for users; the practice behind it is `CHANGES.md`'s 1.0.2 `### Changed`, a
  third-component point release, which retires `Backend.compile_batch` / `link_batch`
  (gh-ocannl-767), collapses `Tensor.raw_unop` / `raw_binop` / `raw_ternop` onto
  `Tensor.raw_accum` plus `Tensor.buffer_of` (gh-ocannl-812), replaces `Tensor.remove_fwd_root` /
  `remove_bprop_root` with `take_forward_code` / `discard_backprop_code`, moves
  `Context.Backends_deprecated.footprint` to `Ir.Low_level.footprint` (gh-ocannl-810), and gives
  `Assignments`, `Indexing`, `Affine`, `Interval`, `Host_inits`, `Compiler_options` and
  `Cpu_topology` explicit interfaces that hide their zero-reference helpers (gh-ocannl-806) — the
  closest precedent for surface nobody calls. None of it went through a compatibility window, and
  the tree carries no `[@@deprecated]` attribute at all (`grep -rn '@@deprecated' lib/ tensor/
  arrayjit/lib/` is empty). Removal is the default, not a rule: surface is retained where keeping
  it costs nothing or serves something other than caller convenience — `lib/ocannl.ml`'s
  backward-compatibility module re-exports, `Parallel.handle.sync_params_to_host` after
  gh-ocannl-333 removed the copying it did, `Operation.centered_uniform1_param_init` and its
  default, which exist to reproduce pre-0.9 random streams. Version depth is no argument against removal either: it tracks release
  scope, not semver (README's Milestones, ROADMAP.md's August 26, 2026 renumbering). The exception
  is a STRING a user typed rather than a name a compiler resolves — `big_models` for `large_models`
  (`arrayjit/lib/utils.ml`), `sync_cc` / `multicore_cc` for `cc` / `multidev_cc`
  (`arrayjit/lib/backends.ml`) — kept as runtime aliases as a matter of course, because a renamed
  key in a stale `ocannl_config` draws `OCANNL warning: unknown config key` and the run CONTINUES
  on the default (`Utils.config_file_args`, quoted in `ocannl_config.reference`), silently changing
  what it computes, where a renamed value constructor fails at the user's compiler with the old
  name in the message. So when a PR finds dead
  exported surface, remove it and say so in the PR body; the `Retired API` changelog line is
  written later, in the editorial pass.
- That argument recurs on removal PRs because it is cheap to raise and, until this bullet, was only
  reconstructible from release practice: staging#742's review raised "deprecate the exported record
  before removing it" as a P2 and cited `AGENTS.md:L17`, which is the Structure and Ownership
  bullet naming `lib/` and states no policy at all. A review citation to a line that does not say
  the thing is a finding about the review — check the citation before complying with it.
- `ocannl_config.reference` ships with every setting COMMENTED OUT, and the two forms are
  load-bearing: a commented-out setting is `#key=value` with NO space after the `#`, while prose
  (and the verbatim profile-payload blocks at the end of the file) always uses `# `. That is how
  `test_config_consistency` tells documented keys from prose, so a new key documented as `# key=…`
  reads as undocumented and fails the test. Config values may not contain `=` — the parser splits
  on it and rejects a line with two, which rules out payload/config values like `-mcpu=native`; a
  setting that needs one gets a word spelling instead (`cc_backend_arch_flags=none`). A value can
  never be the empty string either: empty means "unset" at every source.
- A configuration key has exactly ONE environment spelling, `OCANNL_<KEY>` — gh-ocannl-605 dropped
  the dash-prefixed pair, and gh-ocannl-652 the lowercase `ocannl_<key>`, whose setting is now a
  fatal startup error rather than a silent no-op on case-sensitive environments; native Windows's
  case-INSENSITIVE environment makes `ocannl_backend` the same variable as `OCANNL_BACKEND`, read
  normally (`Utils.env_names_case_insensitive`), while a dashed spelling differs on every platform
  and stays fatal. A dune rule whose output depends on an ambient
  OCANNL setting must declare it as an `(env_var …)` dep — dune tracks nothing a stanza does not
  declare, so an undeclared one leaves a stale target in place and the test green without having
  run. The commandline is the permissive side and always has
  been (`Utils.cmdline_var_names`), but along fixed axes rather than per separator: prefixed or
  not, either case, one leading dash or two (never zero — a bare argument is the host's
  positional), value separator `=`, `_`, `-` or nothing, and the dashing is TWO choices — the
  prefix separator on its own, the key's own separators as a group. So `--ocannl-log_level=1` and
  `--ocannl_log-level=1` are the same setting while a halfway-dashed key
  (`--ocannl-print_decimals-precision=1`) is not a spelling at all. That table
  (`Utils.cmdline_var_prefixes`) is also what the unknown-argument warning matches against — do not
  give it a parser of its own, which is how it came to warn about arguments the reader applied and
  stay silent about ones it ignored.
- A `bin/` tool's own arguments are POSITIONAL and share the commandline with the library's
  `--ocannl_*` settings, so every one of them splits argv the same way — through `Bench_args`
  (`bin/bench_args.ml`, gh-ocannl-634), not a hand-rolled filter. An option is `--`-prefixed or a
  `-` followed by a non-digit, a lone `--` ends the options, and `Bench_args.int` range-checks each
  argument where it is read (`~least:0` for a documented zero); a repeated `--flag=` resolves to the
  FIRST spelling, as `Utils.read_cmdline_var` does. The `--` terminator governs that split only —
  the library scans the whole of `Sys.argv` for its own settings with no terminator and prefix-free
  spellings accepted, so a post-`--` positional spelling a known key (`-- --backend=cuda` as a
  prompt) has already been applied as configuration and cannot be unset from a tool; `Bench_args`
  cannot fix that, so it warns (`shadowing_config`) rather than letting it pass silently. The filter that reads naturally —
  drop everything starting with `-` — is the bug this replaced: it eats a negative extent, and with
  several positionals that shifts every later argument one slot left, so the bench measures a
  geometry nobody asked for and reports a plausible number for it. Review found that same defect in
  two separate copies of the idiom. `test/operations/bench_args_parsing` pins the predicate, the
  slot alignment and the refusals; it links `bench_args` alone, no backend.
- An OCANNL-linked executable's stdout belongs to the program, not to the library: the config
  startup chatter (welcome message, `log_config_sourcing` trace, profile banner) and every other
  library diagnostic go to stderr. That is what lets a tool make stdout a data channel — the
  benchmark one-JSON-line runners, `tools/fit_envelope.exe`'s config-pasteable fits — without
  suppression flags. Keep new library-side reporting on stderr. `test/operations/startup_streams`
  pins both halves: stdout stays empty with the chatter turned all the way up, and a default run's
  stderr is the welcome banner plus whatever warnings there are. The second half is why
  `log_config_sourcing` and `log_level` default to off/0 (gh-ocannl-595): a stream that carries
  eighty lines of routine trace cannot carry a warning, and the unknown-config-key warning is the
  one startup message that means the user made a mistake.
- That contract holds for `.expected` goldens, which diff stdout only, but NOT for inline `%expect`
  blocks: ppx_expect's capture dup2s its temp file onto fd 1 AND fd 2
  (`ppx_expect_runtime_before_test`), so anything on stderr during a block is diffed. The library
  therefore gives the C runtime's `stderr` its own descriptor at startup (`Utils.c_stderr_detached`,
  the `detach_c_stderr` key, gh-ocannl-1031): a foreign C library's diagnostics follow the process's
  real stderr instead of whatever later takes over fd 2, while OCaml's `Stdlib.stderr` keeps writing
  to fd 2 so the library's own reporting is unchanged. It is not filtering — no line is dropped or
  rewritten, the chatter still reaches the real stderr and the run log. The forcing case was ROCm's
  per-event chatter (see the backends note); the general rule is that a golden should never be able
  to acquire a line from a library the program never named.
- Prefer the minimal targeted fix over speculative hardening: offer hardening separately as an
  option with its costs, don't fold it into the fix.
- Git refuses to check out or update a branch that ANOTHER worktree has checked out. This is
  checked-out-branch protection, not `git worktree lock` (which is about pruning and moving a
  worktree, so `git worktree unlock` does nothing for these refusals). While the main checkout holds
  `master`, a linked worktree cannot `git checkout master` ("'master' is already used by worktree
  at …"), `git branch -f master …` ("cannot force update the branch"), or `git fetch origin
  master:master` ("refusing to fetch into branch"). It CAN write the remote ref, which is untouched
  by any of this: `git push origin HEAD:master` — after `git fetch origin && git rebase
  origin/master`, since the push has to fast-forward — lands a commit straight on master from a
  worktree. What that leaves behind is local: the main checkout's `master` is now stale, and only
  that checkout should normally advance it. If NO worktree has `master` checked out, the primary
  form is `git -C <main> fetch origin master:master`; if the main checkout has it checked out, the
  exact complement is `git -C <main> merge --ff-only origin/master` (Git refuses the fetch form
  while any worktree owns `master`). If another worktree owns it, run the merge there instead.
  Either update it, or give every later branch an EXPLICIT start point — `git worktree add -b next
  <path> origin/master`, `git checkout -b next origin/master` — because an omitted start point takes
  the current HEAD, and from a stale main checkout that silently drops the commits just landed.
- The same protection makes `gh pr merge --delete-branch` misleading from a worktree: the merge
  LANDS and only the cleanup fails ("fatal: 'master' is already used by worktree"), so the command
  exits nonzero over an already-merged PR — check the PR's state over REST (`gh api
  repos/<owner>/<repo>/pulls/<n> --jq '"merged=\(.merged) state=\(.state)"'`) before reacting to that
  status, and again before any cleanup. Use REST because `gh pr view --json` and `gh pr checks` ride
  the GraphQL endpoint, which degrades independently of it, and a nonzero `gh pr merge` whose state
  query then 503s is exactly the shape of a merge that DID land; the REST substitute for the checks
  is `gh api repos/<owner>/<repo>/commits/<sha>/check-runs --jq '[.check_runs[]|"\(.name):
  \(.conclusion // .status)"]|.[]'`. `gh pr merge` also returns WITHOUT merging when the base has
  required checks or a merge queue, enabling auto-merge instead, and the steps below would then tear
  down a PR still waiting to land. This repo has neither, so here the flag's own cleanup failure is
  the only way that command misleads.
  Merge without the flag and clean up in this order, anchoring each command as named: `git -C
  <main> push origin --delete <branch>`; `git -C <main> fetch --prune origin`; inspect `git -C <main>
  worktree list --porcelain` for `branch refs/heads/master`, then use the primary `git -C <main>
  fetch origin master:master` if NONE has it checked out, or the exact complement `git -C
  <master-owner> merge --ff-only origin/master` if one does; when the main checkout is off `master`,
  require `git -C <main> merge-base --is-ancestor <branch> master` to pass; `git -C <main> worktree
  remove <path>`; then `git -C <main> branch -d <branch>` when the main checkout is on `master`, or
  `git -C <main> branch -D <branch>` after that explicit ancestry check when it is not (`-D` also
  follows an independently confirmed squash or rebase merge, whose commits are not ancestors of
  `master`). Git refuses the fetch refspec while any worktree owns `master`; the merge form is safe
  because it is anchored to that owner rather than to whichever branch happens to be in `<main>`.
  `git -C <main>` anchors the working directory only; it says nothing about the checked-out branch,
  which is the volatile part during a wave.
  The sequence runs green from inside the worktree it deletes, verified end to end in a scratch
  repo, and both ordering constraints are load-bearing. `-d` tests the branch's UPSTREAM, falling
  back to HEAD only when there is none, so deleting the remote branch FIRST avoids a tautology about
  `origin/<branch>` — left in place, it deletes an unmerged topic with only a warning. With the
  upstream gone it is a real "merged into master" check only when the main checkout's HEAD is
  `master`; off `master`, the explicit `merge-base` command is that safety check and `-D` only
  performs the deletion. Anchoring matters because `worktree remove` deletes the current directory
  when run from inside it, and any later unanchored command dies with "fatal: Unable to read current
  working directory".
- A backend-gated leg must never print a bare `p "<claim>" true` on the backend that cannot run it:
  the golden line is then byte-identical to a verified run's, so neither the transcript nor a
  reviewer can tell the claim was never evaluated (this is how a `Tensorize` leg came to "cover" the
  gh-528 interior-batch bug). Report it with `Verdict.skipped ~backend "claim"`, which prints the
  same stdout line — the golden must stay backend-uniform, and dune's `(test)` stanza diffs stdout
  ONLY, so stderr is free — and announces the skip on stderr. `grep SKIPPED` over a run then
  enumerates exactly what that hardware did not verify. Backend-specific goldens are the wrong tool
  for this: the leg is *absent* on that backend rather than differently-valued, and a golden per
  backend would have to be regenerated on hardware the author usually does not have. The other honest form is putting the
  condition in the label itself (`"… (skipped: non-C backend)"`), which distinguishes the golden
  line; a bare `true` whose label is indistinguishable is the one to reject.
- The skip helper keeps the CLAIM line uniform; it says nothing about the descriptive `printf`s
  beside it. Guarding the whole leg — device run, dump, and claim — behind the `else` branch leaves
  the dump printed on some backends and not others, which is a golden diff on exactly the backend
  that skips. Hoist the descriptive output out of the branch, computing it from something that runs
  everywhere (usually the host-side reference the claim compares against), and gate only the
  device-dependent comparison.
- Executed parity checks need a nonzero guard on the REFERENCE, not just the comparison: a fragment
  mapping that reads outside a staged block, a candidate kernel that never ran, and a reference
  whose own setup collapsed all produce all-zeros, and zeros compare equal to zeros. The convention
  is a file-local `nonzero name a` that raises, applied where each reference array is produced —
  once per producer, not per comparison. Guard the reference side only: a zero candidate against a
  nonzero reference is already a `false` in the golden, which is more diagnosable than an exception.
- A tolerance cannot reject an input-independent forward if the reference itself does not move:
  every leg sits at one constant and every parity line reads `true`. `benchmarks/orchestrate.py`'s
  `loss_moved` is the model; in-tree, require the reference's own spread to exceed the tolerance it
  gates (`mixed_prec_parity`, `precision_policy_parity`) or that distinct inputs give distinct
  outputs (`gpt2_dry_run`'s positions-differ). "All finite" is not such a guard — all-zeros is
  finite.
- A new configuration key must not have an existing key as a NAME PREFIX. `Utils.read_cmdline_var`
  matches an argument against `key ^ ("_" | "-" | "=" | "")` and takes whatever follows as the value,
  so `--ocannl_cc_parallel_grid_private_bytes_cap=N` was read as `cc_parallel_grid` with the value
  `private_bytes_cap=N` and crashed the run — the key was renamed `cc_grid_private_bytes_cap`.
  Nothing checks this: the consistency tests check documentation, classification and read sites, not
  name disjointness.
- Stacked PRs: once the base PR merges, RETARGET the stacked one to master BEFORE merging it. A merge
  into the now-stale base branch lands the work on that branch and nowhere else while GitHub still
  reports the PR as "merged" — staging#168's conv sketches were stranded exactly that way and had to
  be re-landed as staging#170. The same audit question ("is this on master?") is worth asking of any PR whose
  base was not master.
- Cross-repository work references always name their home: `staging#NNN` for PRs in
  `lukstafi/ocannl-staging`, `gh-ocannl-NNN` for upstream issues cited as facts about the codebase,
  and `ahrefs/ocannl#NNN` for other upstream issue mentions. A bare `#NNN` silently resolves against
  whichever repository renders the note, so the agent-notes scan refuses it outside code examples.
  Hashless `PR NNN`/`issue NNN` forms name no repository either and do not belong in the notes.
- For a measurement or report PR, substance stabilizes early: two full review rounds plus one
  verdict-stability check, after which findings are answered rather than actioned unless they touch
  validity, consequence, or arithmetic (validated on the gh-ocannl-530 campaign, where rounds 5–7 were
  framing churn). The failure mode review is actually there to catch runs in both directions — quoting
  whichever control flatters the story — so a report shows ALL matched contrasts side by side rather
  than the decisive one.
- When the next review finding is "leg X missed guard Y", look for the unfactored duplication instead
  of patching leg X. In `tools/sweep.sh`'s nine rounds every point-wise guard had a leg, a path or a
  machine it had not been applied to, and the fixes that actually closed a class REMOVED or UNIFIED
  mechanism: one `flock` replacing a directory-plus-pid-file dance with its reclaim races and
  `kill -0` pid-reuse hole, one `run_capped` replacing three hand-rolled background-and-publish-pid
  call sites.
- Any file OCANNL publishes for a later process to read — a schedule-cache entry, a checkpoint, the
  cc probe cache — goes through `Utils.Atomic_file` (`arrayjit/lib/atomic_file.mli`), never through
  a hand-rolled `<path>.tmp`. Three parts, and a hand-rolled copy usually has one or two: a unique
  staging name (a fixed `.tmp` lets two writers stream into one file and commit a mixture), removal
  on every failing path, and `cleanup_stale` for the writer killed inside its commit window, which
  cannot clean up after itself. The NAME alone has to satisfy three things at once, each of them a
  review finding against an earlier draft: it is recognized whole
  (`<stem>.ocannl-stage.<pid>.<counter>.<nonce>`, the three fields fixed-width hex), because the
  sweep deletes whatever the predicate accepts and a substring match would take out somebody's
  `report.ocannl-stage.backup`; the target contributes a BOUNDED stem (truncation plus a digest, cut
  at a UTF-8 boundary — APFS and Windows both refuse a malformed name), or a long checkpoint name
  pushes the staging name past the 255-byte component limit; and pid-plus-counter is not unique
  across hosts or pid namespaces sharing a filesystem, so a nonce joins them and the file is created
  `O_EXCL` — a taken name is retried, never shared. Generator, recognizer and the `.gitignore` rule
  must describe the SAME set: the fields are fixed-width precisely so a glob can be exact (git reads
  `[0-9]*` as a digit then anything, so a variable-width field hides an ordinary file from its
  author), and the nonce is a full 64-bit draw so no value the recognizer accepts is one the
  generator cannot produce. Names are compared caselessly, since `Model.bin` and
  `model.bin` are one file on Windows and on a default macOS volume. The Windows halves
  are measured, not inferred (gh-ocannl-588): a live mapping blocks neither the rename nor a delete
  but PINS the file's size, so reopening the target for writing is the one operation Windows
  refuses — publish by rename, never by truncation; close the staging file before renaming or
  removing it, since the C runtime opens without `FILE_SHARE_DELETE`; and expect a rename to fail
  transiently while another process holds the target open, which is why the commit retries a
  bounded number of times.

- Bootstrap configuration uses `Utils.resolve_config_value` with a truncated source list
  (gh-ocannl-604): `no_config_file` never consults a file, and bootstrap keys never consult
  profile payloads. Welcome suppression obeys ordinary precedence, including an explicit false
  overriding a file's true. Deferred startup tracing reports the carried `config_source`;
  `resolve_profile_selection` normalizes each source before the same resolver chooses its winner.
