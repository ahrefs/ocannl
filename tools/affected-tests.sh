#!/usr/bin/env bash
# Which per-test aliases can a change reach? Read from dune's own description
# of the build, never from a hand-kept map (gh-ocannl-1235). The entry point is
# `tools/test-run.sh affected`, which refuses while a run holds this worktree
# and then runs this script. It prints aliases and runs none of them.
#
# Usage:
#   tools/test-run.sh affected [--referenced-only] [--gpu-gated] --since BASE [-- PATHSPEC...]
#   tools/test-run.sh affected [--referenced-only] [--gpu-gated] REV | REV..REV | REV...REV [-- PATHSPEC...]
#   tools/test-run.sh affected [--referenced-only] [--gpu-gated] --files PATH...
#   ... [--why ALIAS|all]... (with any of the forms above)
#   ... --harnesses (with any of the forms above, without alias selection or
#   explanation options: the harness commands, below, without asking dune)
#   tools/test-run.sh affected --harness-census [MEMBER] (alone: what each
#   harness reaches, and through which token, below)
#
# `--referenced-only`, `--gpu-gated`, `--why` and `--harnesses` may stand before a
# `--` (after `--since BASE`, a revision, or the `--files` paths too).
#
# `--gpu-gated` filters the ranked reach to aliases whose owner sources name
# binds_hardware_grid or on_gpu, or name leg/in_leg/skipped alongside
# on_cuda/on_hip/on_metal
# or a literal GPU backend, or whose owner/rule has a stanza-local GPU
# ocannl-backend marker, or whose owner reads a declared `; ocannl-backend-default:`
# naming a GPU backend (cuda, hip, metal) whose listed `Module.def` targets match
# the current diff (gh-ocannl-2382).
# Comments are ignored for source tokens; this reads whole files, not scopes or
# gate expressions.
# It may overselect (an unused name, a CPU leg beside a GPU literal) and miss
# computed gates, generated sources or gates hidden in helpers. It does not
# prove backend coverage, and keeps the unfiltered verdict and diagnostics.
# Combine with --referenced-only for the first tier; omit it to retain indirect
# library consumers. This is the GPU-specific minimum: ungated every-backend
# tests in the unfiltered list also exercise GPUs and are excluded here.
# No backend-specific subdivision is inferred.
#
# Backend-default markers (gh-ocannl-2382; closed grammar, fail-closed exit 2):
#   - Definition form inside a `(library)` stanza:
#     `; ocannl-backend-default: <name> = <backend>[,...] <Module.def>[,...] -- <reason>`
#   - Reader form inside a `(test)`, `(tests)`, `(executable)`, `(executables)`,
#     or inline-test `(library)` stanza:
#     `; ocannl-backend-default: <name> -- <reason>`
#   `<name>` matches `[a-z][a-z0-9-]*`, `<backend>` is from `{cc,multidev_cc,cuda,hip,metal}`
#   (no `none`, no empty/duplicate entries, no spaces inside either comma list),
#   `<Module.def>` is `^[A-Z][A-Za-z0-9_']*\.[A-Za-z_][A-Za-z0-9_']*$` (def not `_`),
#   `<reason>` has at least two words after `--` or `—`. Every raw-text occurrence
#   of `ocannl-backend-default:` in a dune file must sit in a valid comment inside
#   a permitted stanza; duplicate definitions and undefined reader names fail.
#   When a reached alias's owner reads a default whose `Module.def` matches `%mod_defs`
#   (that definition changed, or `Module` changed as a whole; never comment-only or
#   sibling-definition edits) and no own rule of that alias pins `; ocannl-backend:`
#   disjoint from the default's backends, the alias is promoted from LINK-ONLY to
#   first-tier REFERENCED, and if the default names a GPU backend it is selected by
#   `--gpu-gated`. Unreached aliases, transitive callees not listed in the definition,
#   and off-build modules are outside this reading.
#
# `--why ALIAS` (gh-ocannl-1711; repeatable, the leading `@` optional) explains
# that alias on stderr, in `affected: why @ALIAS: ...` lines, whether it is
# printed or not, and changes nothing on stdout; `--why all` explains every
# alias listed, link-only ones included. The first line is its verdict: REFERENCED (printed first),
# BLIND (printed first, for an action not known here), LINK-ONLY (printed
# last), NOT LISTED, or no per-test alias. The lines after it are the reasons,
# each naming the changed path behind it: a dune file over its directory, a
# rule of its own that reads a changed path, an alias its rules need; and per
# reached owner, the library chain to the stanza a changed path reached, then
# what made it referenced -- the token naming the changed module, a
# re-export of it (with where its re-exporter qualifies or opens it), or a changed definition, and
# where that token stands (a source of its own, a project ppx's, or a helper
# library's definition, traced back to the token in the owner's code that
# pulled that definition in) -- or why its code counts as not all known, or,
# for LINK-ONLY, the changed parts its code names none of.
#
# The changed paths come from `git diff --name-only` with the given revision
# argument, in git's own forms: one REV compares it with the working tree, so
# uncommitted edits count, and files git does not track yet are added to that
# form. `--since BASE` is that form with REV the merge-base of HEAD and BASE
# (the base branch, e.g. the staging remote's `master`): a branch's own
# changes, whatever BASE gained since it forked. A REV that is not an ancestor
# of HEAD draws a warning, since its own later commits then count as changes.
# There is no default base: remote names are not the contract here.
# `--files` names repository-relative paths directly, without git.
#
# stdout: one alias per line, `@DIR/FAMILY-NAME[-VARIANT]`, ready for
# `tools/test-run.sh run build ...` or `tools/test-run.sh fresh ...`: sorted,
# then the link-only ones, sorted (the ranking is described below);
# `--referenced-only` leaves those off. stderr: what each changed path reached,
# a per-directory count, the link-only aliases, and what this script could
# not place. The contract: LINK-ONLY is a heuristic ranking, never a
# verdict that an alias cannot break. `--referenced-only` is a deliberate
# trade, faster with a possible miss, for iterating mid-change, except the
# explicit CROSS-CUTTING local run: `prepush --referenced-only --since BASE`
# (gh-ocannl-2072), plus exercising tests and applicable slow aliases from the
# full list, link-only ones included. Otherwise the run before a push is `tools/test-run.sh prepush --since BASE`
# (gh-ocannl-1829): the default list, link-only aliases included, without its
# `slow-` and `train-` members (the nightly CI run builds `@train`, no CI job
# runs `@slow`, and no per-PR job runs either, so `prepush` names the `slow-`
# ones and counts the `train-` ones for the caller to run those the change
# exercises) -- unless the verdict, below, is `cross-cutting`: `prepush` then
# refuses, a full-suite run on a CPU-only box before the merge owns the list
# but its `slow-` aliases, and the local run is those plus the tests
# exercising the change (AGENTS.md). The graph is always the CURRENT worktree's, whatever
# revisions were named.
#
# What dune is asked (each answer is checked against the forms listed below,
# and an answer of any other form stops the run, exit 2):
#   - `dune describe workspace --with-pps`: each library and `executables`
#     stanza this build compiles, its modules, and the libraries it requires.
#     It omits `tests` stanzas and leaves preprocessing out of `requires`.
#   - `dune describe external-lib-deps`: every stanza with the project
#     libraries it names, ppx rewriters included (`tests` ones and those this
#     build does not compile too, such as an optional backend library; their
#     modules stay unknown). Its edges are added to the first answer's. A
#     library it names that neither answer lists is skipped when the stanza
#     naming it is one this build does not compile either (the listing gives
#     public names only, and such a stanza may use a local one); from any other
#     stanza it stops the run.
#   - `dune show aliases DIR...` and `dune show targets DIR...`: the aliases
#     and the targets of every directory holding a `dune` file. A `NAME.exe`
#     target no stanza there names is a `tests` stanza naming no library,
#     which neither describe answer lists.
#   - `dune exec --root . --no-build -- printf '%s\n' 'system=%{system}'`
#     supplies the current context's system for an otherwise ownerless
#     module in a directory with literal OS guards. A failed query or an
#     answer outside the one-line system form leaves ordinary mapping in
#     force, including its UNMAPPED fallback.
#   - `dune rules TARGET...`, for each per-test alias's own targets (below),
#     each generated module file, and every C stub object (`X.o`) and stub
#     archive (`libNAME_stubs.a`): the
#     executables those rules run and the files, or globs of files, they read.
#     A generated file such a rule reads (a `copy_files` copy, a file list) is
#     asked in turn, and what its rule reads counts as read; compiled objects
#     and executables are not followed (an executable is reached through its
#     stanza). Queries are batched to stay within any command-line limit.
#     `dune rules` prints a glob with its escapes dropped, so a glob also
#     matches with its braces, brackets and commas read literally, and as its
#     own text. A compiled artifact a rule reads
#     (a stanza's `.NAME.objs` files, an installed library's) makes that
#     stanza one of the rule's owners, like an executable the rule runs; an
#     alias a rule needs (`(deps (alias ...))`) is recorded as well. Only the
#     file `-o` names is read: printing a rule can build what its dependencies
#     are computed from, and those actions' output goes to dune's own streams.
#     An answer of another form is kept in $TMPDIR/affected-dune-rules.PID.txt,
#     and the refusal names the offending atom and its byte offset.
#   - The probe (gh-ocannl-1541): `dune rules @ALIAS` evaluates an alias's
#     actions lazily, which records no dependency, so what an alias rule's own
#     action reads dune never prints. The listings and `dune rules` are
#     therefore asked of a scratch copy of the sources (the probe) in which
#     each dune file is dune's own canonical text (`dune format-dune-file`),
#     each rule attached to an alias that declares no targets is a TWIN -- the
#     same rule writing a file of its own, its aliases kept defined by an
#     `(alias (name X) (deps TWIN))` -- and each `(test)` or `(tests)` name
#     has a golden (an empty one where it has none), so dune makes its output
#     rule. The twin's rule is what the action reads, as dune evaluates it.
#     An alias some stanza makes depend on files (a rule with targets, an
#     `(alias)` stanza) is also asked as `dune rules @ALIAS`, which expands
#     dependencies, helper aliases included. On this repository the probe
#     costs about 10 s a run (its build directory starts cold; the dune cache
#     serves the compiles), and a compared dune edit about 30 s in all.
#     OCANNL_TOOL_AFFECTED_PROBE=0 skips it, at the price of every alias
#     action read blind (and the reach only a twin shows, such as an
#     executable of another name a run-and-diff rule runs); a probe dune does
#     not load falls back the same way, and says so.
#   - For a changed `dune` file, the old probe: the probe with that file at
#     the old revision. Both answer `dune show aliases`/`targets` and `dune
#     rules` for every target of its directory, and their forms (dune's
#     canonical text) are compared too.
#     For these compared files, tests' literal `modules` lists (or their
#     default names) on either side also own preprocessed rules and changed
#     sources. Ordered sets and other nonliteral module forms keep the unknown
#     owner fallback; an unexplained rule still reaches the whole directory.
#
# Boundary, as an allowlist. Whatever it does not cover is reported on stderr,
# never silently dropped. THE CONVERTER IS A STRICT ALLOWLIST: it reads dune's
# canonical text only as top-level forms, their heads, the `alias`/`aliases`/
# `targets`/`target`/`action` fields, literal `name`/`names`, literal
# `include`/`subdir` paths, and the fields described under "Literal
# dependencies". It rewrites targetless rules with literal alias names and
# one action field, and adds dependency witnesses to eligible aliases.
# Other forms pass to Dune verbatim. A form the converter cannot vouch for
# keeps the affected aliases under the fallback, named on stderr.
#   - A per-test alias is `@DIR/FAMILY-NAME` or `@DIR/FAMILY-NAME-VARIANT`
#     that `dune show aliases` lists, FAMILY one of runtest, train, slow. Other
#     families (`@scans`, `@lifecycle`, `@bin-smoke`, ...) are aggregates and
#     never printed.
#   - Its owners are the stanzas of DIR with NAME as a name, as a library's
#     local name or as a module name, plus every executable that a rule of its
#     own targets runs. Its own targets are DIR's targets named `NAME-VARIANT.*`
#     or `NAME.VARIANT.*` (`NAME.*` when there is no variant), or `NAME_*` (a
#     rule-made `NAME_actual.ml` beside `runtest-NAME-ppx`), other than
#     compiled objects, executables and preprocessed copies: `fresh` attributes
#     a test's time by the same names, `NAME_*` aside. A golden alias is one
#     whose own targets, `NAME_*` aside, include an `.expected` file. An alias
#     with neither owners nor own targets is unattributed: it is printed
#     whenever a reached stanza or a changed path lies in its directory, since
#     what it reads is unknown.
#   - A changed `.ml`/`.mli` reaches the stanza whose module it is (by
#     directory and module name: `foo.ml`, `foo.mli`, a `(select)` choice
#     `foo.cudajit.ml` are `Foo`); failing that, in a directory with `tests`
#     stanzas, the one named after it or else every one there, since dune does
#     not describe their modules; failing that, the stanzas of its directory
#     this build does not compile, and the path is UNMAPPED. A module file
#     with no source of its name is generated: any changed path its rule
#     reads (a `.mll`, a `.mly`, a `(select)` choice) reaches its stanza, and
#     so does the executable the rule runs. A changed path a stub object's rule
#     reads (a `.c`, a header) reaches the library whose stub archive bundles
#     that object, or else every executable and `tests` stanza of its
#     directory, since which one links it dune does not say without computing
#     link rules; so does a reached executable the object's rule runs (a
#     generator writing a header it includes). A changed `dune` file not
#     compared across revisions (below), a `dune-project` or `dune-workspace`
#     file, or a fragment a dune file
#     `(include ...)`s, reaches every stanza and every per-test alias of the
#     directories beneath it (an `env` stanza applies there too), and every
#     alias elsewhere whose own targets' rules read a file or glob of those
#     directories (a file their rules make). Reach then spreads to each stanza
#     requiring a reached library, to a fixpoint.
#   - An alias is printed when its directory's `dune` file changed, when an
#     owner is reached, or when a rule of its own targets reads a changed path
#     (a file, or a glob over source files): the repository scans are found
#     this way, and so is a golden, whose copy into the build is one of the
#     own targets of the alias that diffs it. A changed path reaching nothing
#     else is listed as unread.
#   - An alias's own action is not in `dune rules` output. With the probe an
#     alias is DESCRIBED when every action attached to it is a twin or a
#     test's golden diff (dune's documented `(test)` action: its output rule,
#     made in the probe, shows the stanza's `(deps)` and `(action)`): its own
#     targets then include its twins, and it is printed only for what their
#     rules read. It stays BLIND -- an inline-test library's alias (`runtest-LIB`),
#     a directory with a (cram) stanza, an alias a rule the converter
#     does not rewrite attaches to (one with an `(:include)` or a `dynamic-run`,
#     whose dependencies `dune rules` does not print), every alias of a
#     directory holding a form outside the allowlist -- and is printed as
#     without the probe. Without the probe, a golden alias's action diffs its
#     golden against an output a rule makes, both its own targets, whose rules
#     are read: an alias is golden when its own targets hold a `.expected` and a
#     rule-made file beside it. Any other alias's action is opaque -- a `tests`
#     stanza with no golden runs its executable under the stanza's `(deps)`; a
#     rule that runs and diffs in one action owns a golden but no rule-made
#     output (gh-ocannl-1431: taken for a golden pair,
#     `runtest-kernel_attribution` was never listed for the benchmarks file its
#     `(deps)` name). A blind or opaque alias is printed whenever a changed path
#     other than an owned module source lies in its directory or is copied
#     there, and whenever any changed path is no build input (a script, a
#     config, a golden).
#     An alias whose rules need a printed alias is printed too, to a fixpoint:
#     a golden pair's output rule carries the ambient
#     `runtest-env_spelling_gate` for this reason (gh-ocannl-1378) -- in the
#     needed alias's tier, one printed only blind included: a broken needed
#     alias breaks the needing one's build. A needed alias that is no per-test
#     alias (a helper) is an entry of its own, never printed, read as a
#     per-test one; one dune makes itself (`.PKG-files`, what `(package PKG)`
#     needs) is read through `dune rules @ALIAS`, its installed files followed
#     to the build's. Dune 3.25 prints no such alias, but the files a
#     `.packages/HASH` staging area holds for the action: each is read as the
#     shared install area's link it stages, followed the same way.
#   - A changed `dune` file compared across revisions (git forms only; both
#     sides have it; the worktree's copy is the right-hand side's; both probes
#     load) reaches what dune says changed: a rule of its directory that differs
#     between the probes -- a changed twin lists its aliases; a changed `.exe`,
#     `.pp.ml`, object or archive rule reaches its stanza as a whole; any other
#     changed target is read like a changed path, by the aliases' rules, a
#     generated module's, a stub's, and the link and preprocessing rules of
#     every stanza (`link_deps`, `preprocessor_deps`) -- plus a per-test alias
#     new on this side, plus, from the forms that differ, a changed library,
#     executables or tests stanza (compile flags live in rules not asked) by its
#     literal names as dune lists them here, and a changed `(alias)` stanza's
#     per-test alias. A library form whose only differences are its literal
#     `modules` and `libraries` (a split; gh-ocannl-1651) is reached by its
#     modules instead, both sides' (its archive, alias-module and preprocessing
#     rules changing with them), so the link-only ranking reads it. A changed form of any other kind (`env`, `subdir`,
#     `include`, `dirs`, an unknown head, an `(alias)` with an action), a
#     changed rule this reader cannot map to an owner, or a failed comparison
#     keeps the reach below for that file, and says why.
#
# The harness legs (gh-ocannl-1900). The shell harnesses are no dune aliases,
# so what they read is read from their own text, never from dune. Their
# membership is `tools/test-harnesses.sh --list`, CI's sole list of them; each
# changed path reaching one is named on stderr with the reason, and
# `affected: harness: CMD` gives the command that runs what it reaches; the
# closing summary counts those commands when there are any (gh-ocannl-1929:
# `..., and N harness command(s), named above`), and says nothing of them
# when no changed path reaches a harness. With
# `--harnesses` those commands are stdout instead of the aliases, one per line
# (`HARNESS` for every leg, `HARNESS --only 'ERE'` for some), and dune is not
# asked at all. Boundary, as an allowlist:
#   - A harness's PROGRAM is its own file plus every script it loads, to a
#     fixpoint: a script is a file ending `.sh` or `.py`; a shell `.` or
#     `source` command, or a `# shellcheck source=` directive, loads each script
#     its words name; a Python `import M` or `from M import N` loads `M.py`
#     (and `M/N.py`, or `N.py` for a relative `from . import N`).
#   - Text names a path when a maximal run of `[A-Za-z0-9_./-]` in it (trailing
#     dots dropped), or a part of that run after one of its slashes, is the
#     path itself (one holding a `/` or a `.`), its basename (one holding a
#     `.`), or for an OCaml `X.ml` the basename `X.exe`. Lines whose first
#     non-blank character is `#` are comments, and name nothing. A changed
#     path is named whether or not the worktree still has it.
#   - Fixture and heredoc literals in `.sh` files do not count as reaches
#     (gh-ocannl-2380). Closed allowlist of excluded shell forms (everything
#     outside it stays scanned, fail-closed):
#       * Unpiped `cat`/`tee` data heredoc bodies (`<<DELIM` / `<<-DELIM`,
#         never `<<<` here-strings): excluded ONLY when the opener logical
#         command parses cleanly (not inside a single- or multi-line quote,
#         `$(...)`/`<(...)`/`>(...)`/`` `...` `` substitution, or piped
#         compound block), its owning stage has command head literally `cat`
#         or `tee` (ignoring leading `VAR=val` assignments and `env`/`command`/
#         `nice`/`exec`/`nohup`/`builtin`, not overridden by a function or
#         alias in the harness's program unless called via `builtin`/`command`),
#         has no output process substitution (`>(...)`), is the last stage of
#         its pipeline (no downstream pipe stage or trailing `|`), AND when
#         inside a shell function body redirects stdout to a file (`>`, `>|`,
#         `1>`, `1>|`, `&>`, not `/dev/stdout` or `/dev/fd/...` and not later
#         redirected via `>&N`). A
#         quoted delimiter (`<<'EOF'`, `<<"EOF"`, `<<\EOF`) skips the whole
#         body; an unquoted delimiter (`<<EOF`) skips literal body text while
#         still scanning `$(...)` and `` `...` `` command substitutions inside
#         it. Any other heredoc opener (interpreters such as `sh`/`bash`/
#         `python3`/`perl`/`awk`, `source /dev/stdin`, `while read ...; done`,
#         wrappers like `timeout`/`sudo`, variable commands, any piped
#         `cat <<'EOF' | ...` or `{ cat <<'EOF'; } | ...`, `cat > >(bash)`,
#         `tee >(bash)`, unredirected `cat`/`tee` inside a function body, or an
#         unclosed heredoc) stays scanned fail-closed. A
#         harness that writes a real helper script through a data heredoc
#         (e.g. `cat >"$TMP/helper.py" <<'PY'`) declares its repository inputs
#         with `# affected-reads:`.
#       * Literal arguments of unpiped top-level `echo` and `printf` commands
#         (excluded ONLY when `echo`/`printf` is the last stage of its
#         pipeline with no downstream stage, trailing `|`, enclosing piped
#         compound command such as `{ printf ...; } | ...`, or output process
#         substitution `>(...)`, when inside a shell function body redirects
#         stdout to a file as above, not `printf -v`, not overridden by an
#         `echo`/`printf` function or alias in the harness's program unless
#         called via `builtin`/`command`, and not inside `eval`, a single- or
#         multi-line `$(...)`/`<(...)`/`>(...)`/`` `...` `` substitution, or a
#         variable assignment), spanning explicit trailing-`\` continuations
#         or a multiline quoted argument: command substitutions (`$(...)`,
#         `` `...` ``), process substitutions (`<(...)`, `>(...)`), and
#         input/append redirections (`<`, `<>`, `>>`, `&>>`) stay scanned.
#       * Truncating output redirection targets (`>`, `>|`, `&>`; never `<`,
#         `<>`, `>>`, `&>>`, or `>(...)`): the destination path is overwritten
#         from scratch rather than read or appended to; command/process
#         substitutions inside the target stay scanned.
#     Variable assignments (`VAR=...`), non-printing commands (`cat FILE`,
#     `grep`, `cp`, script invocations), process substitutions (`<(...)`,
#     `>(...)`), and Python string literals are never excluded.
#   - When a run names several paths that way (a basename with several owners,
#     gh-ocannl-2342), it names those sharing the most trailing path parts with
#     it (`X.exe` matching `X.ml`); ties keep every owner, so a bare basename
#     names them all. The run's first part is no path part, and is left out of
#     the count, unless the run starts the line or follows a blank, one of
#     `=(:,<>|;&` or `[`, or a quote or backquote that itself starts the line
#     or follows one of those (after `$V`, `${V}`, `)` or a closing quote or
#     backquote it is a fragment); a first part that counts also matches a path part it ends
#     after a character no run holds (a quoted `"fixture data/x.py"`). Never
#     a dropped read: a run `D/B` read from any directory C is the file C/D/B,
#     which ends with all of `D/B`, so it always has the most. Within a
#     harness's program (not in what `loads` follows), a shell `$V/R` or
#     `${V}/R` where a path starts (as above), with V written once in its file as
#     `V=$(cd "$(dirname "$0")[/..]..." && pwd)` (`CDPATH=`, `--`, quotes,
#     `-P` and a `|| ...` tail allowed; `${BASH_SOURCE[0]}` for `$0`) and in no
#     other file of that program, and a `$(dirname "$0")/R` there, name the one
#     owner at R from the harness's directory (`$0` is the harness in each of
#     its program's files) or from the file's own (`BASH_SOURCE`), when R
#     there is one of the owners; otherwise the rule above applies.
#     In a `.py` file of a harness's program (gh-ocannl-2380), a `__file__`-
#     relative path join names the one owner at the joined relative path from
#     that `.py` file's own directory when that owner is among the tied owners
#     (otherwise falling back to the rule above). Closed allowlist of Python
#     forms: `os.path.join(BASE, LIT1 [, LIT2 ...])` and `BASE / LIT1 [/ LIT2
#     ...]` (not followed by another `/`) where each `LIT` is a plain single-
#     or double-quoted relative path literal (`[A-Za-z0-9_./-]+`, no leading
#     `/`) and `BASE` is `os.path.dirname(...)` (nested allowed) around
#     `__file__`, `os.path.abspath(__file__)` or `os.path.realpath(__file__)`,
#     or `(pathlib.)?Path(__file__)(.resolve())?` with `.parent` (repeated) or
#     `.parents[N]` (optional `str(...)` wrapper), or a variable V written once
#     in that `.py` file to one of those forms (no reassignment, parameter
#     shadowing, or rebound `__file__`/`os`/`pathlib`/`Path`). Another module's
#     `mod.__file__`, non-literal join segments, aliased `from os.path import`
#     helpers, a reassigned or inherited variable and every other spelling are
#     not read, so they keep every tied owner.
#   - A harness reaches every path its program's text names, and the program
#     of every script among them (a script it runs, with what that loads); what
#     such a script names in turn is not followed, so a harness that drives a
#     script through another must name it. It reaches its program too.
#   - A harness's header (the comment block `--help` prints, from line 2) may
#     declare legs: `# affected-reads: GLOB... [--only 'ERE']`, each GLOB a
#     repository path in which `*` stands for any run of characters but `/`.
#     `**` is refused: use `*` within one segment and an explicit list of
#     directories for depth. No recursive glob syntax is read;
#     ERE the harness's `--only` selector (no quote, no tab). A changed path
#     the text does not reach, matched by a declaration of a harness, reaches
#     the legs it names (every leg without `--only`): the harness's word on
#     which of its legs read files its code does not name (a table `--stubbed`
#     finds). A declaration only adds: what the text reaches stays every leg's.
#     Any other `affected-reads` line, in the header or out of it, and an ERE
#     bash refuses, stop the run (exit 2); a GLOB matching no file is warned
#     about.
#   - A harness's command is whole when any changed path reaches it whole; else
#     `--only` carries each declared ERE, `(A)|(B)` for several.
#   A module source no stanza of this build lists (not even one it does not
#   compile) that a harness reaches is mapped to those legs, not UNMAPPED.
#   A changed script no harness reaches is named on stderr.
#
# Hand-run shell scripts (gh-ocannl-2413): the harness reader checks every
# visible regular *.sh file for a single marker. The line consists of a
# comment introducer, one space and `harness: none -- REASON` in the
# uninterrupted column-one comment header following the shebang. The marker
# sentinel is reserved wherever it occurs in script text, including after code. The
# script path consists of ASCII letters, digits, underscores, periods and
# hyphens in slash-separated components. The reason is two or more
# space-separated printable ASCII words. A malformed,
# duplicate or misplaced declaration refuses. A marker conflicts with any
# harness reach in the graph above, including declared globs, or a literal
# script basename in a Dune stanza's atoms (including quoted commands and
# existing literal stanza and dependency include files, across guards). Comments and bulk Dune
# glob/source-tree reads do not name a script individually. Generated includes
# and computed filenames are outside the literal reading. On a successful
# ask an exact `affected: PATH: verified hand-run harness marker` line on
# stderr is the planner's evidence for that changed path.
#
# A directory with only library, executable, test, rule or alias stanzas
# guarded by false `(= %{system} LITERAL)` comparisons in Dune's current
# context accounts for an otherwise ownerless module there. Other guards,
# mixed directories, ancestor includes and optional dependency failures keep
# the UNMAPPED path.
# Concrete reads from elsewhere are still selected.
#
# Literal dependencies (gh-ocannl-2304): for each changed `*.sh` path, stderr
# carries one line `affected: PATH: named as a dependency by the rule of
# @ALIAS` per per-test alias that names PATH, sorted, whatever the alias's
# rank. One stanza of a described alias must both name and read PATH. For a
# targetless `(rule)` attached to the alias, the probe reads its twin. For an
# `(alias)` stanza with a literal name and only `name`, `deps`, `enabled_if`,
# `package` or `locks` fields, the probe adds a private target with the same
# deps and guards. `dune rules @ALIAS` must include that stanza's target, and
# its rule must directly read PATH as a File dependency.
# In either case, names are unquoted, literal relative paths inside the tree:
# a deps atom, a `(file ...)` there, either inside a named `(:NAME ...)` group,
# or a `%{dep:PATH}` token (in any rule field or in an alias's deps). A glob,
# a `source_tree`, or a dependency read only by a different alias stanza gives
# no credit. Rules with targets, tests, blind aliases, disabled stanzas, and
# dependencies reported only at runtime are outside this reading.
# `tools/merge-build-check.sh` counts such an alias as the script's coverage
# when its tests tier runs it. Naming is all this reads: a rule that lists a
# script only so a text scan can read it names it as much as one running it.
# For each such alias whose own rules execute `test/operations/sweep_harness_driver.exe`,
# stderr also carries one line `affected: PATH: reached by sweep driver alias @ALIAS`
# (gh-ocannl-2414), which `tools/merge-build-check.sh` reads for the scoped Linux shell leg.
#
# The harness census (gh-ocannl-2342): `affected --harness-census [MEMBER]`
# prints, for every member (or MEMBER alone), what the boundary above reaches,
# read with the same index and steps, whatever changed: one line per reached
# path or declaration, tab-separated `MEMBER PATH STEP WHERE TOKEN OWNERS`,
# sorted bytewise. STEP is `self` (the harness), `loads` or `names` (the step
# that first reached PATH; WHERE is the `FILE:LINE` of the text and TOKEN the
# run that named it, OWNERS how many paths that run named) or `declares`
# (PATH a declared GLOB, TOKEN its `--only 'ERE'` or `-`). `-` fills an empty
# field. It takes no revision, `--since` or `--files`, and none of
# `--harnesses`, `--gpu-gated`, `--referenced-only` and `--why`; a MEMBER not
# in `tools/test-harnesses.sh --list` stops the run. Dune is not asked. Exit 0
# or 2.
#
# The verdict (gh-ocannl-1921): whenever the list is printed (exit 0 or 3), the
# LAST line of stderr is the trailer `affected: verdict=V`, V one of `ok`,
# `cross-cutting`, `incomplete` or `cross-cutting,incomplete`: what a caller
# reads instead of the prose above it (`tools/test-run.sh prepush` does, and
# refuses a stderr without it). `cross-cutting`: the first tier spans over half
# the per-test aliases, or the printed list does though its first tier does not
# (gh-ocannl-1996: a library module most tests only link; `--referenced-only`
# prints the first tier alone, so it is judged on that). CROSS-CUTTING opens
# stderr. Alias queries buffer their other diagnostics until exit, so the
# banner also precedes the alias list when stdout and stderr are combined.
# Its reason lists up to three changed paths with the widest observed
# reach among the counted aliases (first tier when that is wide, otherwise
# the full list). Counts can overlap; they come from the existing reach
# witnesses, not a rerun with each path in isolation. The detailed line below
# the listing says which threshold was crossed and who runs the list;
# `incomplete`: a changed path is UNMAPPED, which is also what exit 3 says.
# Both are decided in one place, `verdict` in the reader below. No trailer on
# exit 2, nor under `--harnesses`, which prints no alias list to judge.
#
# Exit: 0 every changed `.ml`/`.mli` was mapped; 3 the list is printed, but a
# changed `.ml`/`.mli` is UNMAPPED (see stderr; for a module of a library this
# build does not compile, an OFF-BUILD line counts what a build compiling it
# would reach, read from the dune files, and names the aliases whose code names
# it -- a CUDA module on a Mac); 2 usage, git or dune failure, nothing printed.
# `--harnesses` exits 0 or 2.
set -u
die() { echo "affected: $*" >&2; exit 2; }
# One collation everywhere: `sort` and Perl's `sort` then agree bytewise, so
# the printed order never depends on the caller's locale (gh-ocannl-1431).
export LC_ALL=C

CDPATH= cd -P "$(dirname "$0")/.." || die "cannot cd to the repository root"
command -v perl >/dev/null 2>&1 || die "perl not found"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/affected.XXXXXX") || die "cannot create a temporary directory"
tmp=$(cd "$tmp" && pwd -P) || die "cannot resolve $tmp"
trap "exit 130" INT TERM
trap 'rm -rf "${tmp:?}"' EXIT

usage="usage: affected [--referenced-only] [--gpu-gated] [--why ALIAS|all]... [--harnesses] --since BASE | REV | REV..REV | REV...REV [-- PATHSPEC...] | affected [--referenced-only] [--gpu-gated] [--why ALIAS|all]... [--harnesses] --files PATH... | affected --harness-census [MEMBER]"
# Link-only aliases (see the Perl below) are printed last, and named on stderr;
# --referenced-only leaves them off stdout.
referenced_only=0
gpu_gated=0
harnesses=0
census=0 census_member=
: >"$tmp/why"
mode=git
old_rev= new_rev=
revs=()
paths=()
since=
# `--referenced-only` is a flag of the whole call, so it is read in any position
# before a `--` (gh-ocannl-1651): `--since B --referenced-only` and `--files P
# --referenced-only` mean what the leading spelling does. After `--` every word is
# a pathspec.
args=()
while [ $# -gt 0 ]; do
  case $1 in
    --referenced-only) referenced_only=1; shift ;;
    --gpu-gated) gpu_gated=1; shift ;;
    --gpu-gated=*) die "--gpu-gated is a flag and takes no value: $1 ($usage)" ;;
    --harnesses) harnesses=1; shift ;;
    --harnesses=*) die "--harnesses is a flag and takes no value: $1 ($usage)" ;;
    # Its MEMBER is the next word, unless that is an option.
    --harness-census)
      census=1; shift
      case ${1:-} in -* | '') ;; *) census_member=$1; shift ;; esac ;;
    --harness-census=?*) census=1; census_member=${1#--harness-census=}; shift ;;
    --harness-census=) die "--harness-census=: name a member, or give none ($usage)" ;;
    # A flag: with a value it is refused by name, never read as a path.
    --referenced-only=*) die "--referenced-only is a flag and takes no value: $1 ($usage)" ;;
    --why)
      [ $# -ge 2 ] && [ -n "$2" ] || die "--why needs an alias, or all ($usage)"
      printf '%s\n' "$2" >>"$tmp/why"; shift 2 ;;
    --why=?*) printf '%s\n' "${1#--why=}" >>"$tmp/why"; shift ;;
    --why=) die "--why needs an alias, or all ($usage)" ;;
    --) args+=("$@"); break ;;
    *) args+=("$1"); shift ;;
  esac
done
set -- ${args[@]+"${args[@]}"}
# The census reads every member's text, never a change, and prints no alias.
if [ "$census" = 1 ]; then
  [ "$harnesses" = 0 ] || die "--harness-census and --harnesses print different lists ($usage)"
  [ "$gpu_gated" = 0 ] || die "--harness-census prints no alias, so --gpu-gated has none to filter ($usage)"
  [ "$referenced_only" = 0 ] || die "--harness-census prints no alias, so --referenced-only has nothing to leave off ($usage)"
  [ ! -s "$tmp/why" ] || die "--harness-census prints no alias, so --why has none to explain ($usage)"
  [ $# -eq 0 ] || die "--harness-census reads no change, so it takes no revision, --since or --files: $* ($usage)"
  mode=census
# Both rank and explain aliases, which --harnesses does not print.
elif [ "$harnesses" = 1 ]; then
  [ "$gpu_gated" = 0 ] || die "--harnesses and --gpu-gated print different lists ($usage)"
  [ "$referenced_only" = 0 ] || die "--harnesses prints no alias, so --referenced-only has nothing to leave off ($usage)"
  [ ! -s "$tmp/why" ] || die "--harnesses prints no alias, so --why has none to explain ($usage)"
else
  # Reach diagnostics are produced as the graph is read. Give the completed
  # verdict first use of stderr, including on the probe's fallback path.
  exec 3>&2 2>"$tmp/diagnostics" || die "cannot buffer diagnostics"
  trap 'cat "$tmp/diagnostics" >&3; rm -rf "${tmp:?}"' EXIT
fi
if [ "${1:-}" = --files ]; then
  shift
  [ $# -ge 1 ] || die "--files needs at least one path ($usage)"
  mode=files
  paths=("$@")
else
  while [ $# -gt 0 ]; do
    case $1 in
      --files) die "--files comes first, before any revision argument ($usage)" ;;
      --) shift; paths=("$@"); break ;;
      --since)
        [ $# -ge 2 ] || die "--since needs the base branch, e.g. --since <remote>/master ($usage)"
        [ -z "$since" ] || die "--since given twice ($usage)"
        since=$2; shift 2 ;;
      --since=*) [ -z "$since" ] || die "--since given twice ($usage)"; since=${1#--since=}; shift ;;
      -*) die "unknown option: $1 ($usage)" ;;
      *) revs+=("$1"); shift ;;
    esac
  done
  [ -z "$since" ] || [ ${#revs[@]} -eq 0 ] || die "--since BASE and a revision argument are exclusive ($usage)"
  [ ${#revs[@]} -le 1 ] || die "one revision argument at most, got ${#revs[@]} ($usage)"
fi

in_git=0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && in_git=1
if [ "$mode" = census ]; then
  : >"$tmp/changed"
elif [ "$mode" = files ]; then
  printf '%s\n' "${paths[@]}" >"$tmp/changed"
else
  [ "$in_git" = 1 ] || die "not a git checkout; use --files"
  # No default base: which remote carries the base branch is the checkout's
  # business (AGENTS.md: remote names are not the contract), so it is named.
  if [ -n "$since" ]; then
    # The branch's own changes: from where it left BASE, not from BASE's tip,
    # which would also count every commit BASE gained since (gh-ocannl-1379).
    git rev-parse --verify -q "${since}^{commit}" >/dev/null || die "not a revision: $since"
    base=$(git merge-base HEAD "$since") || die "HEAD and $since share no history"
    echo "affected: --since $since: comparing with the merge-base ${base}" >&2
    revs=("$base")
  fi
  [ ${#revs[@]} -eq 1 ] ||
    die "name the base: affected --since <base-branch>, REV, REV..REV, or --files PATH... ($usage)"
  case ${revs[0]} in
    *..*) ;;
    *)
      git rev-parse --verify -q "${revs[0]}^{commit}" >/dev/null || die "not a revision: ${revs[0]}"
      git merge-base --is-ancestor "${revs[0]}" HEAD ||
        echo "affected: warning: ${revs[0]} is not an ancestor of HEAD, so its own later commits count as changes too; --since ${revs[0]} compares with the merge-base instead" >&2 ;;
  esac
  git diff --name-only --no-renames "${revs[0]}" -- ${paths[@]+"${paths[@]}"} >"$tmp/changed" ||
    die "git diff ${revs[0]} failed"
  case ${revs[0]} in
    *..*) ;;
    *) git ls-files --others --exclude-standard -- ${paths[@]+"${paths[@]}"} >>"$tmp/changed" ||
         die "git ls-files failed" ;;
  esac
  # Both sides of each changed `.ml` and `.mli` file: a module edit is ranked by
  # the definitions it touched (the hunks, below). OLD is the revision compared
  # with; NEW the right-hand revision of a range, else the worktree.
  old_rev= new_rev=
  case ${revs[0]} in
    *...*) new_rev=${revs[0]#*...}; old_rev=$(git merge-base "${revs[0]%%...*}" "${new_rev:-HEAD}") ||
             die "no merge-base for ${revs[0]}" ;;
    *..*) old_rev=${revs[0]%%..*}; new_rev=${revs[0]#*..}; old_rev=${old_rev:-HEAD}; new_rev=${new_rev:-HEAD} ;;
    *) old_rev=${revs[0]} ;;
  esac
  while IFS= read -r f; do
    case $f in *.ml | *.mli) ;; *) continue ;; esac
    mkdir -p "$tmp/old/$(dirname -- "$f")" "$tmp/new/$(dirname -- "$f")" || die "cannot create $tmp/old"
    # A file the old side does not have is new: it gets no scoping.
    git show "$old_rev:$f" >"$tmp/old/$f" 2>/dev/null || rm -f "$tmp/old/$f"
    if [ -n "$new_rev" ]; then
      git show "$new_rev:$f" >"$tmp/new/$f" 2>/dev/null || rm -f "$tmp/new/$f"
    elif [ -f "$f" ]; then
      cp -- "$f" "$tmp/new/$f" || die "cannot copy $f"
    fi
  done <"$tmp/changed"
  git diff -U0 --no-color --no-renames --no-ext-diff "${revs[0]}" -- '*.ml' '*.mli' >"$tmp/ml-diff" ||
    die "git diff ${revs[0]} failed"
fi

cat >"$tmp/dune-source.pl" <<'DUNE_SOURCE'
use strict; use warnings;
# A reader for Dune source syntax, as its lexer has it
# (src/dune_sexp/lexer.mll): `;` end-of-line comments only (`#|` and `#;` are
# atom text), quoted strings, block strings, atoms. It dies on text it cannot
# read; the caller decides whether that stops the run. A block string comes
# back OPAQUE (\x01-prefixed: its escapes and its leading space are not
# decoded), which every reader of a stanza's needs takes as unknown.
sub parse_dune {
  my ($s, $what, $markers) = @_;
  my @stack = ([]);
  my $put = sub { push @{$stack[-1]}, $_[0] };
  my $raw_default_occ = $markers ? (() = $s =~ /ocannl-backend-default:/g) : 0;
  my $comment_default_occ = 0;
  pos($s) = 0;
  while (pos($s) < length $s) {
    next if $s =~ /\G\s+/gc;
    if ($s =~ /\G;([^\n]*)/gc) {
      my $comment = $1;
      if ($markers && (index($comment, "ocannl-backend:") >= 0 || index($comment, "ocannl-backend-default:") >= 0)) {
        my $cpos = pos($s) - length($comment) - 1;
        my $line = 1 + (() = substr($s, 0, $cpos) =~ /\n/g);
        $comment_default_occ += (() = $comment =~ /ocannl-backend-default:/g);
        # Defer attribution until the enclosing stanza finishes parsing so comments
        # placed immediately after `(` (before the head atom) are attributed cleanly.
        $put->(["__affected_comment", $comment, $line]);
      }
      next;
    }
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) {
      @stack > 1 or die "unbalanced parentheses in $what at byte " . (pos($s) - 1) . "\n";
      my $l = pop @stack; $put->($l); next;
    }
    if ($s =~ /\G"\\[|>]([^\n]*)/gc) {   # block string: consecutive "\| lines
      my $str = $1;
      $str .= "\n$1" while $s =~ /\G\n[ \t]*"\\[|>]([^\n]*)/gc;
      $put->("\x01" . $str); next;
    }
    if ($s =~ /\G"((?:[^"\\]++|\\.)*+)"/gcs) {
      my $a = $1 =~ s/\\\n[ \t]*//gr;
      $a =~ s/\\(x[0-9A-Fa-f]{2}|[0-9]{3}|.)/length $1 > 1 ? ($1 =~ m{^x} ? chr hex substr($1, 1) : chr $1)
                                            : $1 eq "n" ? "\n" : $1 eq "t" ? "\t" : $1 eq "r" ? "\r" : $1/gse;
      $put->($a); next;
    }
    if ($s =~ /\G([^\s()";]+)/gc) { $put->($1); next }
    die "unreadable text in $what at byte " . pos($s) . "\n";
  }
  @stack == 1 or die "unbalanced parentheses in $what\n";
  if ($markers) {
    if ($raw_default_occ != $comment_default_occ) {
      die "$what: `ocannl-backend-default:` appears outside a comment ($raw_default_occ in text, $comment_default_occ in comments)\n";
    }
    my $is_cm = sub { ref $_[0] && @{$_[0]} && !ref $_[0][0] && $_[0][0] eq "__affected_comment" };
    my $split_reason = sub {
      my ($rest) = @_;
      my ($sep_idx, $sep_len);
      for my $sep ("--", "\xe2\x80\x94") {
        my $idx = index($rest, $sep);
        if ($idx >= 0 && (!defined $sep_idx || $idx < $sep_idx)) {
          $sep_idx = $idx; $sep_len = length($sep);
        }
      }
      return undef unless defined $sep_idx;
      my $decl = substr($rest, 0, $sep_idx) =~ s/^\s+|\s+$//gr;
      my $reason = substr($rest, $sep_idx + $sep_len) =~ s/^\s+|\s+$//gr;
      return ($decl, $reason);
    };
    my $strip_comments;
    $strip_comments = sub {
      my ($node, $acc) = @_;
      return unless ref $node;
      my @kept;
      for my $el (@$node) {
        if ($is_cm->($el)) { push @$acc, [$el->[1], $el->[2]] }
        else { $strip_comments->($el, $acc) if ref $el; push @kept, $el }
      }
      @$node = @kept;
    };
    my $process_top;
    $process_top = sub {
      my ($items) = @_;
      my @kept;
      for my $el (@$items) {
        if ($is_cm->($el)) {
          my ($comment, $line) = ($el->[1], $el->[2]);
          die "$what:$line: `ocannl-backend-default:` sits outside any stanza\n"
            if index($comment, "ocannl-backend-default:") >= 0;
          next;
        }
        if (ref $el) {
          # Strip leading comments inside a form first so `(; comment\n subdir ...)` or
          # `(; comment\n test ...)` reveals its true head atom.
          my @direct_cm;
          while (@$el && $is_cm->($el->[0])) {
            my $c = shift @$el;
            push @direct_cm, [$c->[1], $c->[2]];
          }
          if (@$el >= 2 && !ref $el->[0] && $el->[0] eq "subdir" && !ref $el->[1]) {
            for my $c (@direct_cm) {
              die "$what:$c->[1]: `ocannl-backend-default:` sits outside any stanza\n"
                if index($c->[0], "ocannl-backend-default:") >= 0;
            }
            my @body = @$el[2 .. $#$el];
            $process_top->(\@body);
            @$el = ($el->[0], $el->[1], @body);
            push @kept, $el;
            next;
          }
          my @cm = @direct_cm;
          $strip_comments->($el, \@cm);
          my $head = (@$el && !ref $el->[0]) ? $el->[0] : "";
          for my $c (@cm) {
            my ($comment, $line) = @$c;
            if (index($comment, "ocannl-backend:") >= 0 && $head =~ /^(?:library|executables?|tests?|rule)$/) {
              my $at = index($comment, "ocannl-backend:");
              my $rest = substr($comment, $at + length("ocannl-backend:")) =~ s/^\s+|\s+$//gr;
              my ($decl, $reason) = $split_reason->($rest);
              my $bstr = defined $decl ? $decl : ($rest =~ /^([a-z_]+(?:[ \t]*,[ \t]*[a-z_]+)*)/ ? $1 : "");
              if ($bstr =~ /^[a-z_]+(?:[ \t]*,[ \t]*[a-z_]+)*$/) {
                my @bks = split /[ \t]*,[ \t]*/, $bstr;
                push @$el, ["affected_gpu_backend"] if grep { /^(?:cuda|hip|metal)$/ } @bks;
                push @$el, ["affected_rule_backends", @bks] if $head eq "rule";
                push @$el, ["affected_stanza_backends", @bks] if $head ne "rule";
              }
            }
            if (index($comment, "ocannl-backend-default:") >= 0) {
              my $occ = () = $comment =~ /ocannl-backend-default:/g;
              die "$what:$line: two `ocannl-backend-default:` declarations in one comment\n" if $occ > 1;
              die "$what:$line: `ocannl-backend-default:` sits outside any stanza\n" unless length $head;
              my $trimmed = $comment =~ s/^\s+|\s+$//gr;
              my $at = index($trimmed, "ocannl-backend-default:");
              my $rest = substr($trimmed, $at + length("ocannl-backend-default:")) =~ s/^\s+|\s+$//gr;
              my ($decl, $reason) = $split_reason->($rest);
              die "$what:$line: `ocannl-backend-default:` is missing `--` separating declaration from reason\n"
                unless defined $decl;
              my @rwords = grep { length } split /[ \t]+/, $reason;
              die "$what:$line: `ocannl-backend-default:` reason must be at least two words\n"
                if @rwords < 2;
              if (index($decl, "=") >= 0) {
                die "$what:$line: `ocannl-backend-default:` expected a single `=` in definition\n"
                  if (() = $decl =~ /=/g) > 1;
                my ($lhs, $rhs) = split /=/, $decl, 2;
                $lhs =~ s/^\s+|\s+$//g; $rhs =~ s/^\s+|\s+$//g;
                die "$what:$line: `$lhs` is not a valid backend-default name\n"
                  unless $lhs =~ /^[a-z][a-z0-9-]*$/;
                my @toks = grep { length } split /[ \t]+/, $rhs;
                die "$what:$line: `ocannl-backend-default:` expected `<backend>[,...] <Module.def>[,...]` after `=`\n"
                  unless @toks == 2;
                my @bks = split /,/, $toks[0], -1;
                my %bseen;
                for my $b (@bks) {
                  die "$what:$line: invalid backend `$b` in `ocannl-backend-default:`\n"
                    unless $b =~ /^(?:cc|multidev_cc|cuda|hip|metal)$/;
                  die "$what:$line: duplicate backend `$b` in `ocannl-backend-default:`\n"
                    if $bseen{$b}++;
                }
                my @tgs = split /,/, $toks[1], -1;
                my %tseen;
                for my $t (@tgs) {
                  die "$what:$line: invalid target `$t` in `ocannl-backend-default:`\n"
                    unless $t =~ /^([A-Z][A-Za-z0-9_']*)\.([A-Za-z_][A-Za-z0-9_']*)$/ && $2 ne "_";
                  die "$what:$line: duplicate target `$t` in `ocannl-backend-default:`\n"
                    if $tseen{$t}++;
                }
                die "$what:$line: `ocannl-backend-default:` definition belongs on a `(library)` stanza, not `($head)`\n"
                  unless $head eq "library";
                push @$el, ["affected_default_def", $lhs, join(",", @bks), join(",", @tgs), "$what:$line"];
              } else {
                die "$what:$line: `$decl` is not a valid backend-default name\n"
                  unless $decl =~ /^[a-z][a-z0-9-]*$/;
                die "$what:$line: `ocannl-backend-default:` reader belongs on a test, executable, or inline-test library stanza, not `($head)`\n"
                  unless $head =~ /^(?:tests?|executables?|library)$/;
                push @$el, ["affected_default_read", $decl, "$what:$line"];
              }
            }
          }
        }
        push @kept, $el;
      }
      @$items = @kept;
    };
    $process_top->($stack[0]);
  }
  return $stack[0];
}
1;
DUNE_SOURCE

# The harness legs (gh-ocannl-1900; the header's boundary): their membership
# from CI's own list, every file their text may name, and harness.pl's reading
# of that text. Dune is not asked: `--harnesses` stops here.
for x in harness-cmds harness-err harness-read harness-eres; do : >"$tmp/$x"; done
if [ -f tools/test-harnesses.sh ]; then
  bash tools/test-harnesses.sh --list >"$tmp/harness-members" 2>"$tmp/err" ||
    die "tools/test-harnesses.sh --list failed:
$(cat "$tmp/err")"
  if [ "$in_git" = 1 ]; then
    git ls-files -z --cached --others --exclude-standard >"$tmp/files" || die "git ls-files failed"
  else
    find . \( -name _build -o -name .git \) -prune -o -type f -print0 >"$tmp/files" ||
      die "cannot list files"
  fi
  cat >"$tmp/harness.pl" <<'HARNESS'
# harness.pl TMP: the harness legs the changed paths reach (the header's
# boundary). Reads TMP/changed, TMP/files and TMP/harness-members; writes
# harness-cmds (one command per line), harness-err (lines for stderr),
# harness-read (`PATH<TAB>N` for each changed path N harnesses read) and
# harness-eres (`ERE<TAB>HARNESS:LINE` for every declared selector).
use strict; use warnings;
use Cwd ();
my ($tmp) = @ARGV;
require "$tmp/dune-source.pl";
sub fail { print STDERR "affected: $_[0]\n"; exit 2 }
sub lines {
  my ($f) = @_;
  open my $h, "<", $f or fail("cannot read $f");
  my @l = grep { $_ ne "" } map { s/\r?\n\z//r } <$h>;
  close $h;
  return @l;
}
my %once;
my @changed = grep { !$once{$_}++ } lines("$tmp/changed");
my @members = lines("$tmp/harness-members");
-f $_ or fail("tools/test-harnesses.sh --list names $_, which is not a file here") for @members;
sub inventory {
  open my $fh, "<", "$tmp/files" or fail("cannot read $tmp/files");
  local $/ = "\0";
  my @paths = map { chomp; s{\A\./}{}r } <$fh>;
  close $fh or fail("cannot close $tmp/files");
  return @paths;
}
my %known;
$known{$_} = 1 for inventory(), @changed, @members;
my @known = sort keys %known;
# The names text may know a path by: itself (holding a `/` or a `.`), its
# basename (holding a `.`), and `X.exe` for an `X.ml`.
my %idx;
for my $p (@known) {
  my $b = $p =~ s{.*/}{}r;
  $idx{$p}{$p} = 1 if $p =~ m{[/.]};
  $idx{$b}{$p} = 1 if $b =~ /\./;
  $idx{"$1.exe"}{$p} = 1 if $b =~ /\A(.+)\.ml\z/;
}
sub is_script { $_[0] =~ /\.(?:sh|py)\z/ }
sub dir_of { $_[0] =~ m{\A(.*)/} ? $1 : "" }
sub norm {   # a relative path with its `.` and `..` parts folded; undef above the root
  my @o;
  for my $s (split m{/}, $_[0]) {
    next if $s eq "" || $s eq ".";
    if ($s eq "..") { return undef unless @o; pop @o } else { push @o, $s }
  }
  return join "/", @o;
}
# How many trailing parts path P shares with the parts S. Unless FRAG, the
# first part may end a part of P after a character no run holds: the line
# may have split a quoted name there (`"fixture data/x.py"`).
sub shared {
  my ($p, $frag, @s) = @_;
  my @q = split m{/}, $p;
  my $n = 0;
  while (@s && @q) {
    my ($x, $y) = (pop @s, pop @q);
    last unless $x eq $y || ($n == 0 && $x =~ /\A(.+)\.exe\z/ && $y eq "$1.ml")
      || (!$frag && !@s && $x ne "" && $y =~ m{[^A-Za-z0-9_./-]\Q$x\E\z});
    $n++;
  }
  return $n;
}
# A run of path characters -> the paths it, or a part after one of its
# slashes, names; of several, those sharing the most trailing parts with it,
# or the one at its anchor (the header's rule). FRAG: its first part is no
# path part; AT: the directory its anchor stands for.
sub resolve {
  my ($t, $frag, $at) = @_;
  $t =~ s/\.+\z//;
  my @s = split m{/}, $t, -1;
  my %r;
  while ($t ne "") {
    $r{$_} = 1 for keys %{$idx{$t} // {}};
    $t =~ s{\A[^/]*/}{} or last;
  }
  my @c = sort keys %r;
  return @c if @c < 2;
  shift @s if $frag;
  if (defined $at) {
    my $p = norm(join "/", $at, @s);
    return ($p) if defined $p && $r{$p};
  }
  my %n = map { $_ => shared($_, $frag, @s) } @c;
  my ($most) = sort { $b <=> $a } values %n;
  return grep { $n{$_} == $most } @c;
}
# The runs of path characters in a line, each [RUN, FRAG, ANCHOR]: FRAG when
# its first part continues text before it (the header's list of what starts
# one); ANCHOR `$0` or `$BASH_SOURCE` after `$(dirname "$0")` or its
# `BASH_SOURCE` form, or V after `$V` or `${V}`, when that expansion starts a
# path and a `/` starts what follows it.
my $self_arg = qr{"?(?:\$0|\$\{0\}|\$BASH_SOURCE|\$\{BASH_SOURCE(?:\[0\])?\})"?};
sub starts_path { $_[0] eq "" || $_[0] =~ /[\s=(:,<>|;&\[]\z/ || $_[0] =~ /(?:\A|[\s=(:,<>|;&\[])["'`]\z/ }
sub runs {
  my ($line) = @_;
  my @r;
  while ($line =~ m{([A-Za-z0-9_./-]+)}g) {
    # Every key in %idx holds `.` or `/`; skip bare words before slicing $line.
    next unless index($1, ".") >= 0 || index($1, "/") >= 0;
    my ($t, $pre) = ($1, substr($line, 0, $-[1]));
    my ($frag, $anchor) = (1, undef);
    if ($pre =~ /\$\z/) {
      $anchor = $1 if $t =~ m{\A([A-Za-z_]\w*)/} && starts_path(substr($pre, 0, -1));
    } elsif ($pre =~ /\$\(dirname\s+(?:--\s+)?($self_arg)\)\z/) {
      my ($arg, $at) = ($1, $-[0]);
      $anchor = $arg =~ /BASH_SOURCE/ ? '$BASH_SOURCE' : '$0' if $t =~ m{\A/} && starts_path(substr($pre, 0, $at));
    } elsif ($pre =~ /\$\{([A-Za-z_]\w*)\}\z/) {
      my ($v, $at) = ($1, $-[0]);
      $anchor = $v if $t =~ m{\A/} && starts_path(substr($pre, 0, $at));
    } elsif (starts_path($pre)) {
      $frag = 0;
    }
    push @r, [$t, $frag, $anchor];
  }
  return @r;
}
my (%raw, %names, %loads, %assigns, %selfdirs, %sh_base, %sh_parsed, %py_info);
sub raw {
  my ($f) = @_;
  return $raw{$f} if exists $raw{$f};
  my $s = "";
  if (-f $f && open my $h, "<", $f) { local $/; $s = <$h> // ""; close $h }
  return $raw{$f} = $s;
}
# Replace non-newline characters with spaces, preserving physical line numbers.
sub blank_keep_nl { my ($s) = @_; $s =~ s/[^\n]/ /g; return $s }
# Advance past a balanced shell substitution starting at byte offset $i in $s
# ($open is '$(', '${', '<(', '>(', or '`'). Returns the end offset just past
# the closing delimiter, or undef if unbalanced (fail closed).
sub sh_skip_balanced {
  my ($s, $i, $open, $in_dq) = @_;
  my $len = length $s;
  if ($open eq '`') {
    my ($j, $esc) = ($i + 1, 0);
    while ($j < $len) {
      my $c = substr($s, $j, 1);
      if ($esc) { $esc = 0; $j++; next }
      if ($c eq "\\") { $esc = 1; $j++; next }
      return $j + 1 if $c eq '`';
      $j++;
    }
    return undef;
  }
  if ($open eq '${') {
    my ($j, $depth, $isq, $idq, $iesc) = ($i + 2, 1, 0, 0, 0);
    while ($j < $len && $depth > 0) {
      my $c = substr($s, $j, 1);
      if ($iesc) { $iesc = 0; $j++; next }
      if ($c eq "\\" && !$isq) { $iesc = 1; $j++; next }
      if (!$in_dq && $c eq "'" && !$idq) { $isq = !$isq; $j++; next }
      if ($c eq '"' && !$isq) { $idq = !$idq; $j++; next }
      if (!$isq && substr($s, $j, 2) eq '$(') {
        my $nj = sh_skip_balanced($s, $j, '$(');
        return undef unless defined $nj;
        $j = $nj;
        next;
      }
      if (!$isq && !$idq) {
        if ($c eq "{") { $depth++ }
        elsif ($c eq "}") { $depth-- }
      }
      $j++;
    }
    return $depth == 0 ? $j : undef;
  }
  # Stack-balanced scan for $(...), <(...), >(...), tracking nested `case ... esac`
  # so an unparenthesized case pattern terminator `x)` never closes the substitution.
  my @stk = ("sub");
  my ($j, $isq, $idq, $iesc) = ($i + 2, 0, 0, 0);
  while ($j < $len && @stk) {
    my $c = substr($s, $j, 1);
    if ($iesc) { $iesc = 0; $j++; next }
    if ($c eq "\\" && !$isq) { $iesc = 1; $j++; next }
    if ($c eq "'" && !$idq) { $isq = !$isq; $j++; next }
    if ($c eq '"' && !$isq) { $idq = !$idq; $j++; next }
    if (!$isq && substr($s, $j, 2) eq '$(') {
      my $nj = sh_skip_balanced($s, $j, '$(');
      return undef unless defined $nj;
      $j = $nj;
      next;
    }
    if (!$isq && !$idq) {
      if ($c eq "(") { push @stk, "paren" }
      elsif ($c eq ")") {
        pop @stk if $stk[-1] eq "paren" || $stk[-1] eq "sub";
      } elsif (($j == $i + 2 || substr($s, $j - 1, 1) =~ /[\s;|&(]/) &&
               substr($s, $j) =~ /\A(case|esac)(?=[\s;|&()]|\z)/) {
        my $kw = $1;
        if ($kw eq "case") { push @stk, "case" }
        elsif ($stk[-1] eq "case") { pop @stk }
        $j += length($kw);
        next;
      }
    }
    $j++;
  }
  return !@stk ? $j : undef;
}
# Extract only executed command/process substitutions ($(...), `...`, and when
# $respect_sq is true, <(...) / >(...)) plus assignment-bearing parameter
# expansions (${var:=word} / ${var=word}) from an excluded shell literal region,
# replacing everything else (except `\n`) with spaces. When $respect_sq is true
# (shell words, not unquoted heredoc bodies), single-quoted spans suppress
# substitution. Unbalanced delimiters fail closed.
sub extract_cmdsubs {
  my ($s, $respect_sq) = @_;
  # An unescaped backslash-newline inside an unquoted heredoc or double-quoted
  # span can split a substitution opener (`$\` + `(cmd)`); fail closed.
  return $s if $s =~ /(?<!\\)(?:\\\\)*\\\n/;
  my $len = length $s;
  my $out = blank_keep_nl($s);
  my ($sq, $dq, $esc, $i) = (0, 0, 0, 0);
  while ($i < $len) {
    my $ch = substr($s, $i, 1);
    if ($esc) { $esc = 0; $i++; next }
    if ($ch eq "\\" && !$sq) { $esc = 1; $i++; next }
    if ($respect_sq && $ch eq "'" && !$dq) { $sq = !$sq; $i++; next }
    if ($respect_sq && $ch eq '"' && !$sq) { $dq = !$dq; $i++; next }
    my $two = substr($s, $i, 2);
    if ((!$sq && $two eq '$(') || ($respect_sq && !$sq && !$dq && ($two eq '<(' || $two eq '>('))) {
      my $nj = sh_skip_balanced($s, $i, $two);
      if (defined $nj) {
        # Keep the substitution delimiters ($(...), <(...), >(...)) so `loads_of`
        # still sees a command boundary before `source`/`.` inside the substitution.
        substr($out, $i, $nj - $i) = substr($s, $i, $nj - $i);
        $i = $nj;
        next;
      }
      # Unbalanced substitution: fail closed on the remainder.
      substr($out, $i) = substr($s, $i);
      last;
    }
    if (!$sq && $two eq '${') {
      my $nj = sh_skip_balanced($s, $i, '${', (!$respect_sq || $dq) ? 1 : 0);
      if (defined $nj) {
        my $inner = substr($s, $i + 2, $nj - $i - 3);
        if ($inner =~ /\A[A-Za-z_]\w*(?:\[[^\]]*\])?:?=/) {
          substr($out, $i, $nj - $i) = substr($s, $i, $nj - $i);
        } else {
          substr($out, $i + 2, length($inner)) = extract_cmdsubs($inner, $dq ? 0 : $respect_sq);
        }
        $i = $nj;
        next;
      }
      substr($out, $i) = substr($s, $i);
      last;
    }
    if (!$sq && $ch eq '`') {
      my $nj = sh_skip_balanced($s, $i, '`');
      if (defined $nj) {
        substr($out, $i, $nj - $i) = substr($s, $i, $nj - $i);
        $i = $nj;
        next;
      }
      substr($out, $i) = substr($s, $i);
      last;
    }
    $i++;
  }
  return $out;
}
# Unquote a simple shell word (stripping '...', "...", \c).
sub sh_unquote {
  my ($w) = @_;
  my ($out, $sq, $dq, $esc, $i) = ("", 0, 0, 0, 0);
  my $len = length $w;
  while ($i < $len) {
    my $c = substr($w, $i, 1);
    if ($esc) { $out .= $c; $esc = 0; $i++; next }
    if ($c eq "\\" && !$sq) { $esc = 1; $i++; next }
    if ($c eq "'" && !$dq) { $sq = !$sq; $i++; next }
    if ($c eq '"' && !$sq) { $dq = !$dq; $i++; next }
    $out .= $c;
    $i++;
  }
  return $out;
}
my %sh_kw = map { $_ => 1 } qw(if elif then else fi do done while until for case in esac ! time);
# Tokenize a shell string into [type, start, end, text], preserving offsets.
# Returns (undef) if quotes or substitutions are unbalanced (fail closed).
sub sh_tokenize {
  my ($s) = @_;
  my @toks;
  my ($i, $len) = (0, length $s);
  while ($i < $len) {
    pos($s) = $i;
    if ($s =~ /\G[ \t\r\n]+/gc) { $i = pos($s); next }
    if ($s =~ /\G(#[^\n]*)/gc) {
      # Inline comment to end of physical line.
      push @toks, ["comment", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G((?:\d+)?(?:<&|>&))/gc) {
      push @toks, ["redir_fd", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G((?:\d+)?<<<)/gc) {
      push @toks, ["redir_herestr", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G((?:\d+)?<<-?)/gc) {
      push @toks, ["redir_hd", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G((?:\d+)?<>)/gc) {
      push @toks, ["redir_in", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G(&>>|(?:\d+)?>>)/gc) {
      push @toks, ["redir_append", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G(&>|(?:\d+)?>\||(?:\d+)?>)(?!\()/gc) {
      push @toks, ["redir_out", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G((?:\d+)?<)(?!\()/gc) {
      push @toks, ["redir_in", $i, pos($s), $1];
      $i = pos($s);
      next;
    }
    if ($s =~ /\G(&&|\|\||;;&|;&|;;|\|&|[;&|()])/gc) {
      my $op = $1;
      my $tp = ($op eq "|" || $op eq "|&") ? "pipe"
        : $op eq "(" ? "open_grp"
        : $op eq ")" ? "close_grp" : "sep";
      push @toks, [$tp, $i, pos($s), $op];
      $i = pos($s);
      next;
    }
    # Normal shell word (including process substitutions <(...) and >(...)).
    my $st = $i;
    my ($sq, $dq, $esc) = (0, 0, 0);
    while ($i < $len) {
      if ($esc) { $esc = 0; $i++; next }
      pos($s) = $i;
      if ($sq) {
        if ($s =~ /\G[^']+/gc) { $i = pos($s); next }
        $sq = 0; $i++; next;
      }
      if ($dq) {
        if ($s =~ /\G[^"\\\$`]+/gc) { $i = pos($s); next }
      } else {
        if ($s =~ /\G[^ \t\r\n;&|()<>\\'"\$`]+/gc) { $i = pos($s); next }
      }
      my $ch = substr($s, $i, 1);
      if ($ch eq "\\") { $esc = 1; $i++; next }
      if ($ch eq "'" && !$dq) { $sq = 1; $i++; next }
      if ($ch eq '"') { $dq = !$dq; $i++; next }
      my $two = substr($s, $i, 2);
      if ($two eq '${' || $two eq '$(') {
        my $nj = sh_skip_balanced($s, $i, $two, $dq);
        return undef unless defined $nj;
        $i = $nj;
        next;
      }
      if (!$dq && ($two eq '<(' || $two eq '>(')) {
        my $nj = sh_skip_balanced($s, $i, $two);
        return undef unless defined $nj;
        $i = $nj;
        next;
      }
      if ($ch eq '`') {
        my $nj = sh_skip_balanced($s, $i, '`');
        return undef unless defined $nj;
        $i = $nj;
        next;
      }
      last if !$dq && $ch =~ /[ \t\r\n;&|()<>]/;
      $i++;
    }
    return undef if $sq || $dq;
    my $w = substr($s, $st, $i - $st);
    my $tp = $w eq "{" ? "open_grp" : $w eq "}" ? "close_grp" : "word";
    push @toks, [$tp, $st, $i, $w];
  }
  return \@toks;
}
# True if shell word $w contains an unquoted output process substitution `>(`.
sub word_has_out_procsub {
  my ($w) = @_;
  return 0 unless index($w, ">(") >= 0;
  my ($sq, $dq, $esc, $i, $len) = (0, 0, 0, 0, length $w);
  while ($i < $len) {
    my $c = substr($w, $i, 1);
    if ($esc) { $esc = 0; $i++; next }
    if ($c eq "\\" && !$sq) { $esc = 1; $i++; next }
    if ($c eq "'" && !$dq) { $sq = !$sq; $i++; next }
    if ($c eq '"' && !$sq) { $dq = !$dq; $i++; next }
    return 1 if !$sq && !$dq && substr($w, $i, 2) eq ">(";
    $i++;
  }
  return 0;
}
# True if any word or redirection target in $stage holds an output process substitution `>(...)`.
sub stage_has_out_procsub {
  my ($stage) = @_;
  for my $it (@$stage) {
    if ($it->[0] eq "word") {
      return 1 if word_has_out_procsub($it->[3]);
    } elsif ($it->[0] =~ /\Aredir_(?:out|append|in)\z/ && defined $it->[2]) {
      return 1 if word_has_out_procsub($it->[2][3]);
    }
  }
  return 0;
}
# True if shell word $raw is a plain file target (closed allowlist: `/dev/null`
# or a non-`/dev/`, non-`/proc/`, non-`fd/<N>` path with no `>(...)`, and not a
# bare variable `$OUT` / `${OUT}` unless $allow_bare_var is true).
sub is_safe_file_target {
  my ($raw, $allow_bare_var) = @_;
  my $t = sh_unquote($raw);
  return 1 if $t eq "/dev/null";
  return 0 if $t eq "";
  return 0 if $t =~ m{\A/(?:dev|proc)(?:/|\z)};
  return 0 if $t =~ m{(?:/|\A)fd/\d+\z};
  return 0 if word_has_out_procsub($raw);
  return 0 if !$allow_bare_var && $t =~ /\A(?:\$[A-Za-z_]\w*|\$\{[^}]*\})\z/;
  return 1;
}
# True if $stage redirects stdout (fd 1) directly to a plain file (never a
# bare variable, `/dev/*` other than `/dev/null`, `/proc/*`, `>(...)`, or a
# later fd dup like `>&2`/`>&3`).
sub stage_has_file_stdout_redir {
  my ($stage) = @_;
  my $to_file = 0;
  for my $it (@$stage) {
    if ($it->[0] eq "redir_fd") {
      my $op = $it->[1][3];
      $to_file = 0 if $op =~ /\A(?:1?>&|>&)\z/;
    } elsif ($it->[0] eq "redir_out") {
      next unless defined $it->[2];
      my $op = $it->[1][3];
      next unless $op =~ /\A(?:&>|1?>\|?)\z/;
      $to_file = is_safe_file_target($it->[2][3], 0) ? 1 : 0;
    } elsif ($it->[0] eq "redir_append" && defined $it->[2]) {
      my $op = $it->[1][3];
      $to_file = 0 if $op =~ /\A(?:&>>|1?>>)\z/;
    }
  }
  return $to_file;
}
# True if $stage redirects stdout to an unsafe target (e.g. `/dev/fd/3`,
# `/proc/self/fd/3`, or a bare variable when persistent `exec` opened an fd).
sub stage_has_unsafe_stdout_redir {
  my ($stage, $overridden) = @_;
  my $allow_bare = ($overridden && ($overridden->{"\0exec_any_fd"} || $overridden->{"\0exec_stdout"})) ? 0 : 1;
  for my $it (@$stage) {
    if ($it->[0] eq "redir_out" || $it->[0] eq "redir_append") {
      next unless defined $it->[2];
      next unless $it->[1][3] =~ /\A(?:&>|1?>\|?|&>>|1?>>)\z/;
      return 1 unless is_safe_file_target($it->[2][3], $allow_bare);
    }
  }
  return 0;
}
# True if the tokens immediately following a closed compound command at $close_idx
# pipe its output onward (`|` / `|&`) or redirect it into an output process
# substitution `>(...)`, a non-stdout descriptor (`>&3`, `>&2`, `>&$fd`), or an
# unsafe stdout target (`/dev/fd/N`, `/proc/...`, bare variable `$OUT`).
sub after_group_is_piped {
  my ($toks, $close_idx) = @_;
  my $j = $close_idx + 1;
  while ($j < @$toks) {
    my $tp = $toks->[$j][0];
    if ($tp eq "comment") { $j++; next }
    return 1 if $tp eq "pipe";
    if ($tp eq "redir_fd") {
      my $op = $toks->[$j][3];
      my $tgt = ($j + 1 < @$toks && $toks->[$j + 1][0] eq "word") ? $toks->[++$j] : undef;
      if ($op =~ /\A(?:1?>&|>&)\z/) {
        return 1 unless defined $tgt;
        my $raw = $tgt->[3];
        return 1 if word_has_expansion($raw) || word_has_out_procsub($raw);
        my $dst = sh_unquote($raw);
        return 1 unless $dst eq "-" || $dst eq "1";
      }
      $j++;
      next;
    }
    if ($tp =~ /\Aredir_(?:out|append|in|hd|herestr)\z/) {
      my $op = $toks->[$j][3];
      my $tgt = ($j + 1 < @$toks && $toks->[$j + 1][0] eq "word") ? $toks->[++$j] : undef;
      if (defined $tgt) {
        return 1 if word_has_out_procsub($tgt->[3]);
        if (($tp eq "redir_out" || $tp eq "redir_append") && $op =~ /\A(?:&>|1?>\|?|&>>|1?>>)\z/) {
          return 1 unless is_safe_file_target($tgt->[3], 0);
        }
      }
      $j++;
      next;
    }
    last;
  }
  return 0;
}
# Walk compound-command delimiters (`{`/`}`, `(`/`)`, `if`/`fi`, `for`/`while`/`until`/`done`,
# `case`/`esac`) and function headers (`name() {`, `function name {`, `name() (`) across $toks
# using stack $stk and scalar ref $pending_fn. Invokes $on_piped when a closed compound command
# is piped onward or redirected to an executable/unsafe destination, and $on_fn when a function body closes.
sub sh_walk_groups {
  my ($toks, $stk, $pending_fn, $line_idx, $on_piped, $on_fn) = @_;
  my $cmd_pos = 1;
  my $ti = 0;
  while ($ti < @$toks) {
    my $t = $toks->[$ti];
    my ($tp, $w) = ($t->[0], $t->[3]);
    if ($tp eq "comment") { $ti++; next }
    if ($tp eq "sep" || $tp eq "pipe") { $cmd_pos = 1; $$pending_fn = 0; $ti++; next }
    if ($tp =~ /\Aredir_(?:out|append|in|hd|herestr|fd)\z/) {
      $ti++ if $ti + 1 < @$toks && $toks->[$ti + 1][0] eq "word";
      $ti++;
      next;
    }
    if ($tp eq "open_grp") {
      my $is_fn = $$pending_fn ? 1 : 0;
      $$pending_fn = 0;
      push @$stk, [$w, $line_idx, $ti, $is_fn];
      $cmd_pos = 1;
      $ti++;
      next;
    }
    if ($tp eq "close_grp") {
      $$pending_fn = 0;
      if ($w eq ")" && @$stk && $stk->[-1][0] eq "case") {
        $cmd_pos = 1;
        $ti++;
        next;
      }
      my $want = $w eq "}" ? "{" : "(";
      if (@$stk && $stk->[-1][0] eq $want) {
        my $top = pop @$stk;
        $on_fn->($top->[1], $top->[2], $line_idx, $ti) if $top->[3] && $on_fn;
        $on_piped->($top->[1], $top->[2], $line_idx, $ti) if after_group_is_piped($toks, $ti);
      }
      $cmd_pos = 0;
      $ti++;
      next;
    }
    if ($tp eq "word") {
      if ($cmd_pos) {
        if ($w eq "function" && $ti + 1 < @$toks && $toks->[$ti + 1][0] eq "word") {
          $ti += 2;
          if ($ti + 1 < @$toks && $toks->[$ti][0] eq "open_grp" && $toks->[$ti][3] eq "(" &&
              $toks->[$ti + 1][0] eq "close_grp" && $toks->[$ti + 1][3] eq ")") {
            $ti += 2;
          }
          $$pending_fn = 1;
          next;
        }
        if ($ti + 2 < @$toks && $toks->[$ti + 1][0] eq "open_grp" && $toks->[$ti + 1][3] eq "(" &&
            $toks->[$ti + 2][0] eq "close_grp" && $toks->[$ti + 2][3] eq ")") {
          $ti += 3;
          $$pending_fn = 1;
          next;
        }
        my $is_fn = $$pending_fn ? 1 : 0;
        $$pending_fn = 0;
        if ($w =~ /\A[A-Za-z_]\w*\+?=/) { $ti++; next }
        if ($w eq "if") { push @$stk, ["if", $line_idx, $ti, $is_fn]; $ti++; next }
        if ($w eq "for" || $w eq "while" || $w eq "until" || $w eq "select") { push @$stk, ["loop", $line_idx, $ti, $is_fn]; $cmd_pos = 0; $ti++; next }
        if ($w eq "case") { push @$stk, ["case", $line_idx, $ti, $is_fn]; $cmd_pos = 0; $ti++; next }
        if ($w eq "then" || $w eq "else" || $w eq "elif" || $w eq "do" || $w eq "!" || $w eq "time") { $ti++; next }
        if ($w eq "fi" || $w eq "done" || $w eq "esac") {
          my $want = $w eq "fi" ? "if" : $w eq "done" ? "loop" : "case";
          if (@$stk && $stk->[-1][0] eq $want) {
            my $top = pop @$stk;
            $on_fn->($top->[1], $top->[2], $line_idx, $ti) if $top->[3] && $on_fn;
            $on_piped->($top->[1], $top->[2], $line_idx, $ti) if after_group_is_piped($toks, $ti);
          }
          $cmd_pos = 0;
          $ti++;
          next;
        }
      } else {
        $$pending_fn = 0;
      }
      $cmd_pos = 0;
    }
    $ti++;
  }
}
# Split a token list into pipelines -> stages -> [tokens] (redirections bound to
# their target word as [redir_type, op_tok, target_tok]). Pipelines inside a
# piped compound command carry a trailing_pipe sentinel; stages inside a
# single-line function body carry an in_fn sentinel.
sub sh_pipelines {
  my ($toks) = @_;
  my @piped_tok = (0) x scalar(@$toks);
  my @fn_tok    = (0) x scalar(@$toks);
  my @gstk;
  my $pfn = 0;
  sh_walk_groups($toks, \@gstk, \$pfn, 0,
    sub { my (undef, $sti, undef, $eti) = @_; $piped_tok[$_] = 1 for $sti .. $eti },
    sub { my (undef, $sti, undef, $eti) = @_; $fn_tok[$_]    = 1 for $sti .. $eti });
  my (@pipes, @cur_pipe, @cur_stage);
  my ($saw_pipe, $grp_piped, $stage_fn) = (0, 0, 0);
  my $flush_stage = sub {
    if (@cur_stage) {
      push @cur_stage, ["in_fn"] if $stage_fn;
      push @cur_pipe, [@cur_stage];
      @cur_stage = ();
      $stage_fn = 0;
    }
  };
  my $flush_pipe = sub {
    $flush_stage->();
    if (($saw_pipe || $grp_piped) && @cur_pipe) { push @cur_pipe, [["trailing_pipe"]] }
    $saw_pipe = 0;
    $grp_piped = 0;
    if (@cur_pipe) { push @pipes, [@cur_pipe]; @cur_pipe = () }
  };
  my $i = 0;
  while ($i < @$toks) {
    my $t = $toks->[$i];
    if ($t->[0] eq "comment") { $i++; next }
    if ($t->[0] eq "sep" || $t->[0] eq "open_grp" || $t->[0] eq "close_grp") { $flush_pipe->(); $i++; next }
    if ($t->[0] eq "pipe") { $flush_stage->(); $saw_pipe = 1; $i++; next }
    $saw_pipe = 0;
    $grp_piped = 1 if $piped_tok[$i];
    $stage_fn  = 1 if $fn_tok[$i];
    if ($t->[0] =~ /\Aredir_(?:out|append|in|hd|herestr|fd)\z/) {
      my $target = ($i + 1 < @$toks && $toks->[$i + 1][0] eq "word") ? $toks->[++$i] : undef;
      push @cur_stage, [$t->[0], $t, $target];
      $i++;
      next;
    }
    # Leading shell keywords start a new command segment.
    if (!@cur_stage && $t->[0] eq "word" && $sh_kw{$t->[3]}) { $i++; next }
    push @cur_stage, $t;
    $i++;
  }
  $flush_pipe->();
  return \@pipes;
}
# Return ($head_idx, $head_raw, $bypass_fn) of the command word in a stage's
# items, skipping redirections, leading VAR=val assignments, and simple wrappers.
sub stage_command_head {
  my ($stage) = @_;
  my ($i, $bypass_fn) = (0, 0);
  while ($i < @$stage) {
    my $it = $stage->[$i];
    return (undef, undef, 0) if $it->[0] eq "trailing_pipe";
    if ($it->[0] =~ /\Aredir_/ || $it->[0] eq "in_fn") { $i++; next }
    my $w = $it->[3];
    if ($sh_kw{$w}) { $i++; next }
    if ($w =~ /\A[A-Za-z_]\w*\+?=/) { $i++; next }
    my $u = sh_unquote($w);
    if ($u eq "builtin") { $bypass_fn = 1; $i++; next }
    if ($u eq "exec" || $u eq "nohup") { $i++; next }
    if ($u eq "command") {
      $bypass_fn = 1;
      $i++;
      while ($i < @$stage && $stage->[$i][0] eq "word") {
        my $fl = sh_unquote($stage->[$i][3]);
        if ($fl eq "--") { $i++; last }
        if ($fl eq "-p") { $i++; next }
        return (undef, undef, 0) if $fl =~ /\A-/;
        last;
      }
      next;
    }
    if ($u eq "env") {
      $i++;
      while ($i < @$stage && $stage->[$i][0] eq "word") {
        my $nw = sh_unquote($stage->[$i][3]);
        if ($nw eq "--" || $nw eq "-") { $i++; last }
        if ($nw eq "-u" || $nw eq "--unset" || $nw eq "-C" || $nw eq "--chdir" || $nw eq "-S" || $nw eq "--split-string") {
          $i += 2;
          next;
        }
        if ($nw eq "-i" || $nw eq "-0" || $nw eq "--ignore-environment" || $nw eq "--null" ||
            $nw =~ /\A--(?:unset|chdir|split-string)=/ || $nw =~ /\A[A-Za-z_]\w*=/) {
          $i++;
          next;
        }
        # Unknown env option (might consume the next word as its operand): fail closed.
        return (undef, undef, 0) if $nw =~ /\A-/;
        last;
      }
      next;
    }
    if ($u eq "nice") {
      $i++;
      while ($i < @$stage && $stage->[$i][0] eq "word") {
        my $nw = sh_unquote($stage->[$i][3]);
        if ($nw eq "--") { $i++; last }
        if ($nw eq "-n" || $nw eq "--adjustment") { $i += 2; next }
        if ($nw =~ /\A(?:-n\S+|--adjustment=\S+|-\d+)\z/) { $i++; next }
        return (undef, undef, 0) if $nw =~ /\A-/;
        last;
      }
      next;
    }
    return ($i, $w, $bypass_fn);
  }
  return (undef, undef, 0);
}
# True if shell word $w contains an unescaped `$` or `` ` `` outside single quotes.
sub word_has_expansion {
  my ($w) = @_;
  return 0 unless $w =~ /[\$`]/;
  my ($sq, $dq, $esc, $i, $len) = (0, 0, 0, 0, length $w);
  while ($i < $len) {
    my $c = substr($w, $i, 1);
    if ($esc) { $esc = 0; $i++; next }
    if ($c eq "\\" && !$sq) { $esc = 1; $i++; next }
    if ($c eq "'" && !$dq) { $sq = !$sq; $i++; next }
    if ($c eq '"' && !$sq) { $dq = !$dq; $i++; next }
    return 1 if !$sq && ($c eq '$' || $c eq '`');
    $i++;
  }
  return 0;
}
# True if $stage redirects stdout into any custom fd > 2 (`>&3`, `>& 3`, `>& '3'`,
# `>&10`, `1>&12`, `>&$fd`, or bash `>&file`), or into stderr (`>&2` / `1>&2`) when
# persistent `exec 2>...` is present.
sub stage_has_custom_fd_out {
  my ($stage, $overridden) = @_;
  for my $it (@$stage) {
    next unless $it->[0] eq "redir_fd" && $it->[1][3] =~ /\A(?:1?>&|>&)\z/;
    return 1 unless defined $it->[2];
    my $raw = $it->[2][3];
    return 1 if word_has_expansion($raw) || word_has_out_procsub($raw);
    my $dst = sh_unquote($raw);
    next if $dst eq "-" || $dst eq "1";
    return 1 if $dst !~ /\A\d+\z/;
    return 1 if $dst != 2;
    return 1 if $overridden && $overridden->{"\0exec_stderr"};
  }
  return 0;
}
# Closed allowlist for data heredocs: return 1 ONLY when the heredoc at byte
# offset $hd_pos on opener logical command $line belongs to an unpiped `cat` or
# `tee` (the last stage of its pipeline, with no output process substitution
# `>(...)`, custom-fd redirect, or unsafe stdout/tee file operand such as
# `/dev/fd/3`, redirecting stdout to a file if inside a shell function body or
# after a persistent `exec` stdout redirect, and not overridden by a
# function/alias unless via `builtin`/`command`).
# Everything else returns 0 (fail closed).
sub heredoc_is_data {
  my ($line, $hd_pos, $overridden, $in_fn) = @_;
  my $toks = sh_tokenize($line);
  return 0 unless $toks;   # unparseable or inside unclosed $( -> fail closed
  my $pipes = sh_pipelines($toks);
  for my $pipe (@$pipes) {
    my $owner = -1;
    for my $k (0 .. $#$pipe) {
      for my $it (@{$pipe->[$k]}) {
        if ($it->[0] eq "redir_hd" && $it->[1][1] <= $hd_pos && $hd_pos < $it->[1][2]) {
          $owner = $k;
          last;
        }
      }
      last if $owner >= 0;
    }
    next if $owner < 0;
    return 0 if $owner != $#$pipe;   # piped, trailing `|`, or piped compound group -> fail closed
    my $stage = $pipe->[$owner];
    return 0 if stage_has_out_procsub($stage)
      || stage_has_custom_fd_out($stage, $overridden)
      || stage_has_unsafe_stdout_redir($stage, $overridden);
    my $stage_fn = $in_fn || (grep { $_->[0] eq "in_fn" } @$stage) || ($overridden && $overridden->{"\0exec_stdout"}) ? 1 : 0;
    return 0 if $stage_fn && !stage_has_file_stdout_redir($stage);
    my ($hidx, $head, $bypass_fn) = stage_command_head($stage);
    return 0 unless defined $head;
    my $hu = sh_unquote($head);
    return 0 unless $hu eq "cat" || $hu eq "tee";
    return 0 if !$bypass_fn && $overridden && $overridden->{$hu};
    if ($hu eq "tee") {
      my $end_opts = 0;
      for my $j ($hidx + 1 .. $#$stage) {
        my $it = $stage->[$j];
        next if $it->[0] =~ /\Aredir_/ || $it->[0] eq "in_fn";
        my $raw_a = $it->[3];
        my $au = sh_unquote($raw_a);
        if (!$end_opts && $au eq "--") { $end_opts = 1; next }
        if (!$end_opts && $au =~ /\A-/) {
          return 0 if word_has_expansion($raw_a);
          return 0 unless $au =~ /\A(?:-[ai]+|--append|--ignore-interrupts)\z/;
          next;
        }
        $end_opts = 1;
        return 0 unless is_safe_file_target($raw_a, 0);
      }
    }
    return 1;
  }
  return 0;
}
# Filter one logical shell line (possibly spanning multiple lines via `\` or a
# multiline quoted string, preserving `\n`): blank truncating output redirection
# targets (`>`, `>|`, `&>`), heredoc delimiters, and literal arguments of
# unpiped top-level echo/printf (unless overridden by a function/alias, or
# inside a function body / after persistent `exec` stdout redirect without a
# per-command file stdout redirection).
sub filter_sh_logical {
  my ($s, $overridden, $in_fn, $toks) = @_;
  $toks //= sh_tokenize($s);
  return $s unless $toks;   # unbalanced -> fail closed
  my $needs_filter = 0;
  for my $t (@$toks) {
    my $tp = $t->[0];
    if ($tp eq "redir_hd" || $tp eq "redir_out") { $needs_filter = 1; last }
    if ($tp eq "word") {
      my $w = $t->[3];
      if ($w eq "echo" || $w eq "printf"
          || ($w =~ /['"\\]/ && do { my $u = sh_unquote($w); $u eq "echo" || $u eq "printf" })) {
        $needs_filter = 1; last;
      }
    }
  }
  return $s unless $needs_filter;
  my $out = $s;
  my $pipes = sh_pipelines($toks);
  for my $pipe (@$pipes) {
    for my $k (0 .. $#$pipe) {
      my $stage = $pipe->[$k];
      next if $stage->[0][0] eq "trailing_pipe";
      # 1. Blank heredoc delimiters and truncating output redirection targets (> / >| / &>).
      for my $it (@$stage) {
        if ($it->[0] eq "redir_hd" && defined $it->[2]) {
          my ($st, $en) = ($it->[2][1], $it->[2][2]);
          substr($out, $st, $en - $st) = blank_keep_nl(substr($s, $st, $en - $st));
        } elsif ($it->[0] eq "redir_out" && defined $it->[2]) {
          my ($st, $en, $w) = ($it->[2][1], $it->[2][2], $it->[2][3]);
          substr($out, $st, $en - $st) = extract_cmdsubs($w, 1);
        }
      }
      # 2. Blank literal arguments of unpiped echo / printf (closed allowlist:
      #    must be the last stage of its pipeline, not inside a piped compound
      #    group, no output process substitution `>(...)` or custom-fd redirect,
      #    if inside a function body or after persistent `exec` stdout redirect
      #    must redirect stdout to a file, not printf -v or expanded option,
      #    and not overridden by a function/alias unless via `builtin`/`command`).
      my ($hidx, $hraw, $bypass_fn) = stage_command_head($stage);
      next unless defined $hidx;
      my $hu = sh_unquote($hraw);
      next unless $hu eq "echo" || $hu eq "printf";
      next if $k != $#$pipe;
      next if stage_has_out_procsub($stage)
        || stage_has_custom_fd_out($stage, $overridden)
        || stage_has_unsafe_stdout_redir($stage, $overridden);
      my $stage_fn = $in_fn || (grep { $_->[0] eq "in_fn" } @$stage) || ($overridden && $overridden->{"\0exec_stdout"}) ? 1 : 0;
      next if $stage_fn && !stage_has_file_stdout_redir($stage);
      next if !$bypass_fn && $overridden && $overridden->{$hu};
      if ($hu eq "printf") {
        my $has_v = 0;
        for my $j ($hidx + 1 .. $#$stage) {
          my $it = $stage->[$j];
          next if $it->[0] =~ /\Aredir_/ || $it->[0] eq "in_fn";
          my $raw_a = $it->[3];
          my $au = sh_unquote($raw_a);
          last if $au eq "--";
          if ($au =~ /\A-v/ || word_has_expansion($raw_a)) { $has_v = 1; last }
          last if $au !~ /\A-/;
        }
        next if $has_v;
      }
      for my $j ($hidx + 1 .. $#$stage) {
        my $it = $stage->[$j];
        next if $it->[0] =~ /\Aredir_/ || $it->[0] eq "in_fn";
        my ($st, $en, $w) = ($it->[1], $it->[2], $it->[3]);
        substr($out, $st, $en - $st) = extract_cmdsubs($w, 1);
      }
    }
  }
  return $out;
}
# Detect function/alias overrides of echo / printf / cat / tee and persistent
# stdout-redirecting bare `exec` statements (`exec > >(bash)`, `exec >&3`, ...)
# on a shell line.
sub sh_scan_overrides {
  my ($line, $funcs_ref, $toks) = @_;
  return if $line =~ /^\s*#/;
  if ($line =~ /\b(?:echo|printf|cat|tee)\b/) {
    $funcs_ref->{$1} = 1 while $line =~ /(?:^|[;&|({])\s*(?:function\s+)?(echo|printf|cat|tee)\s*\(\s*\)/g;
    $funcs_ref->{$1} = 1 while $line =~ /(?:^|[;&|({])\s*function\s+(echo|printf|cat|tee)\b/g;
  }
  $toks //= sh_tokenize($line);
  if ($toks) {
    my $maybe_ov = 0;
    for my $t (@$toks) {
      next unless $t->[0] eq "word";
      my $w = $t->[3];
      if ($w eq "alias" || $w eq "exec"
          || ($w =~ /['"\\]/ && do { my $u = sh_unquote($w); $u eq "alias" || $u eq "exec" })) {
        $maybe_ov = 1; last;
      }
    }
    return unless $maybe_ov;
    for my $pipe (@{sh_pipelines($toks)}) {
      for my $stage (@$pipe) {
        my ($hidx, $hraw) = stage_command_head($stage);
        if (defined $hidx && sh_unquote($hraw) eq "alias") {
          my $end_opts = 0;
          for my $j ($hidx + 1 .. $#$stage) {
            my $it = $stage->[$j];
            next if $it->[0] =~ /\Aredir_/ || $it->[0] eq "in_fn";
            my $au = sh_unquote($it->[3]);
            if (!$end_opts && $au eq "--") { $end_opts = 1; next }
            next if !$end_opts && $au =~ /\A-/;
            $end_opts = 1;
            $funcs_ref->{$1} = 1 if $au =~ /\A(echo|printf|cat|tee)=/;
          }
        } elsif (!defined $hidx || $hraw =~ /\A\{[A-Za-z_]\w*\}\z/) {
          my ($has_exec, $has_stdout_redir, $has_stderr_redir, $has_any_fd_redir) = (0, 0, 0, 0);
          for my $it (@$stage) {
            if ($it->[0] eq "word" && sh_unquote($it->[3]) eq "exec") { $has_exec = 1 }
            elsif ($it->[0] eq "redir_out") {
              my $op = $it->[1][3];
              $has_any_fd_redir = 1;
              $has_stdout_redir = 1 if $op =~ /\A(?:&>|1?>\|?)\z/ && !defined($hidx);
              $has_stderr_redir = 1 if $op =~ /\A(?:&>|2>\|?)\z/;
            } elsif ($it->[0] eq "redir_append") {
              my $op = $it->[1][3];
              $has_any_fd_redir = 1;
              $has_stdout_redir = 1 if $op =~ /\A(?:&>>|1?>>)\z/ && !defined($hidx);
              $has_stderr_redir = 1 if $op =~ /\A(?:&>>|2>>)\z/;
            } elsif ($it->[0] eq "redir_fd") {
              my $op = $it->[1][3];
              $has_any_fd_redir = 1;
              $has_stdout_redir = 1 if $op =~ /\A(?:1?>&|>&)\z/ && !defined($hidx);
              $has_stderr_redir = 1 if $op =~ /\A2>&\z/;
            }
          }
          if ($has_exec) {
            $funcs_ref->{"\0exec_stdout"} = 1 if $has_stdout_redir;
            $funcs_ref->{"\0exec_stderr"} = 1 if $has_stderr_redir;
            $funcs_ref->{"\0exec_any_fd"} = 1 if $has_any_fd_redir;
          }
        }
      }
    }
  } else {
    while ($line =~ /(?:^|[;&|({])\s*alias\b([^;&|)]*)/g) {
      my $args = $1;
      $funcs_ref->{$1} = 1 while $args =~ /(?:^|\s)['"]?(echo|printf|cat|tee)=/g;
    }
    $funcs_ref->{"\0exec_stdout"} = 1 if $line =~ /(?:^|[;&|({])\s*exec\s+(?:&>|1?>|&>>|1?>>|>&)/;
    $funcs_ref->{"\0exec_stderr"} = 1 if $line =~ /(?:^|[;&|({])\s*exec\s+(?:&>|2>|&>>|2>>|2>&)/;
    $funcs_ref->{"\0exec_any_fd"} = 1 if $line =~ /(?:^|[;&|({])\s*exec\s+(?:\{[A-Za-z_]\w*\}>|&>|\d*>|&>>|\d*>>|\d*>&)/;
  }
}
# Override-independent Pass 1a (heredoc spans) and Pass 1b (multiline
# substitutions, piped compound blocks, function bodies, local overrides, and
# single-line token classification), cached once per `.sh` file.
sub sh_parse_base {
  my ($f) = @_;
  return $sh_base{$f} if $sh_base{$f};
  my @l = split /\n/, raw($f), -1;
  my @kind = ("code") x scalar(@l);
  my %line_toks;
  my (@queue, @closed_hds);
  # Pass 1a: identify heredoc line spans (opener, body, closing delimiter),
  # joining explicit `\`-newline continuations into one logical opener line
  # (so both `bash \\\n <<'SH'` and `cat <<'SH' \\\n | bash` are seen whole)
  # and carrying multiline quote state so quoted text never opens a heredoc.
  my ($sq, $dq) = (0, 0);
  my $i = 0;
  while ($i <= $#l) {
    my $line = $l[$i];
    if (@queue) {
      my $hd = $queue[0];
      my $chk = $line;
      $chk =~ s/^\t+// if $hd->{strip};
      if ($chk eq $hd->{delim}) {
        $kind[$_] = "hd_body" for @{$hd->{body}};
        $kind[$i] = "hd_delim";
        push @closed_hds, $hd;
        shift @queue;
      } else {
        push @{$hd->{body}}, $i;
      }
      $i++;
      next;
    }
    if (!$sq && !$dq && ($line =~ /^\s*#/ || $line !~ /['"\\`\$<#]/)) { $i++; next }
    my ($cur, $log, $pos) = ($i, $line, 0);
    my @new_hds;
    while (1) {
      my ($esc, $len) = (0, length $log);
      while ($pos < $len) {
        if ($esc) { $esc = 0; $pos++; next }
        pos($log) = $pos;
        if ($sq) {
          if ($log =~ /\G[^']+/gc) { $pos = pos($log); next }
          $sq = 0; $pos++; next;
        }
        if ($dq) {
          if ($log =~ /\G[^"\\\$`]+/gc) { $pos = pos($log); next }
        } else {
          if ($log =~ /\G[^'"\\\$`<#]+/gc) { $pos = pos($log); next }
        }
        my $ch = substr($log, $pos, 1);
        if ($ch eq "\\") { $esc = 1; $pos++; next }
        if ($ch eq "'" && !$dq) { $sq = 1; $pos++; next }
        if ($ch eq '"') { $dq = !$dq; $pos++; next }
        if (!$dq && $ch eq "#" && ($pos == 0 || substr($log, $pos - 1, 1) =~ /[\s;|&(]/)) {
          $pos = $len;
          last;
        }
        my $two = substr($log, $pos, 2);
        if ($two eq '${' || $two eq '$(') {
          my $nj = sh_skip_balanced($log, $pos, $two, $dq);
          if (defined $nj) { $pos = $nj; next }
          $pos += 2;
          next;
        }
        if ($ch eq '`') {
          my $nj = sh_skip_balanced($log, $pos, '`');
          if (defined $nj) { $pos = $nj; next }
          $pos++;
          next;
        }
        if (!$dq && $two eq "<<") {
          if (substr($log, $pos, 3) eq "<<<") {
            $pos++ while $pos < $len && substr($log, $pos, 1) eq "<";
            next;
          }
          my $hd_pos = $pos;
          $pos += 2;
          my $strip = 0;
          if ($pos < $len && substr($log, $pos, 1) eq "-") { $strip = 1; $pos++ }
          $pos++ while $pos < $len && substr($log, $pos, 1) =~ /[ \t]/;
          my ($delim, $quoted, $dsq, $ddq, $desc) = ("", 0, 0, 0, 0);
          while ($pos < $len) {
            my $dc = substr($log, $pos, 1);
            if ($desc) { $delim .= $dc; $desc = 0; $pos++; next }
            if ($dc eq "\\" && !$dsq) { $quoted = 1; $desc = 1; $pos++; next }
            if ($dc eq "'" && !$ddq) { $quoted = 1; $dsq = !$dsq; $pos++; next }
            if ($dc eq '"' && !$dsq) { $quoted = 1; $ddq = !$ddq; $pos++; next }
            last if !$dsq && !$ddq && $dc =~ /[\s;&|()<>#]/;
            $delim .= $dc;
            $pos++;
          }
          if ($delim ne "" && !$dsq && !$ddq && !$desc) {
            push @new_hds, { opener => $i, opener_end => $cur, delim => $delim, strip => $strip, quoted => $quoted,
                             op_text => "", op_pos => $hd_pos, body => [] };
          }
          next;
        }
        $pos++;
      }
      if ($esc && $cur < $#l) {
        substr($log, -1, 1, "");
        $pos = length $log;
        $cur++;
        $log .= $l[$cur];
        next;
      }
      last;
    }
    for my $hd (@new_hds) {
      $hd->{op_text} = $log;
      $hd->{opener_end} = $cur;
      push @queue, $hd;
    }
    $i = $cur + 1;
  }
  # Pass 1b: track multiline substitutions ($(...), <(...), >(...), `...`),
  # multiline piped compound blocks, multiline shell function bodies, and
  # function/alias overrides across code lines.
  my @unfilt = (0) x scalar(@l);
  my @in_fn  = (0) x scalar(@l);
  my (%funcs, @blk_stk, @sub_stk);
  my ($msq, $mdq, $in_bt, $pending_fn) = (0, 0, 0, 0);
  $i = 0;
  while ($i <= $#l) {
    if ($kind[$i] ne "code") { $i++; next }
    my $line = $l[$i];
    $in_fn[$i] = 1 if grep { $_->[3] } @blk_stk;
    if (!$msq && !$mdq && !@sub_stk && !$in_bt) {
      if ($line =~ /^\s*#/) { $i++; next }
      # Join explicit backslash-newline continuations into one logical command
      # before discovering overrides (`alias \\\n  cat=bash`, `exec \\\n > >(bash)`)
      # and walking compound groups.
      my $j = $i;
      my $log = $line;
      while ($j < $#l && $kind[$j + 1] eq "code" && $log =~ /(?<!\\)(?:\\\\)*\\\z/) {
        $log =~ s/\\\z/ /;
        $j++;
        $in_fn[$j] = 1 if grep { $_->[3] } @blk_stk;
        $log .= $l[$j];
      }
      my $toks = sh_tokenize($log);
      sh_scan_overrides($log, \%funcs, $toks);
      if ($toks) {
        if ($j == $i && $line !~ /(?<!\\)(?:\\\\)*\\\z/) {
          my $nf = 0;
          for my $t (@$toks) {
            my $tp = $t->[0];
            if ($tp eq "redir_hd" || $tp eq "redir_out") { $nf = 1; last }
            if ($tp eq "word") {
              my $w = $t->[3];
              if ($w eq "echo" || $w eq "printf"
                  || ($w =~ /['"\\]/ && do { my $u = sh_unquote($w); $u eq "echo" || $u eq "printf" })) {
                $nf = 1; last;
              }
            }
          }
          $line_toks{$i} = $nf ? $toks : 0;
        }
        sh_walk_groups($toks, \@blk_stk, \$pending_fn, $i,
          sub {
            my ($sln, undef, $eln, undef) = @_;
            if ($sln < $eln) { $unfilt[$_] = 1 for $sln .. $eln }
          });
        $i = $j + 1;
        next;
      }
    } else {
      $unfilt[$i] = 1 if @sub_stk || $in_bt;
    }
    # Character-level state update for multiline quotes or multiline substitutions.
    my ($esc, $p, $len) = (0, 0, length $line);
    while ($p < $len) {
      my $c = substr($line, $p, 1);
      if ($esc) { $esc = 0; $p++; next }
      if ($c eq "\\" && !$msq) { $esc = 1; $p++; next }
      if ($in_bt) {
        if ($c eq '`') { $in_bt = 0 }
        $p++;
        next;
      }
      if ($c eq "'" && !$mdq) { $msq = !$msq; $p++; next }
      if ($c eq '"' && !$msq) { $mdq = !$mdq; $p++; next }
      if (!$msq && !$mdq && $c eq "#" && ($p == 0 || substr($line, $p - 1, 1) =~ /[\s;|&(]/)) { last }
      my $two = substr($line, $p, 2);
      if (!$msq && ($two eq '$(' || (!$mdq && ($two eq '<(' || $two eq '>(')))) {
        my $nj = sh_skip_balanced($line, $p, $two);
        if (defined $nj) { $p = $nj; next }
        push @sub_stk, ["sub", $mdq];
        $mdq = 0;
        $unfilt[$i] = 1;
        $p += 2;
        next;
      }
      if (!$msq && $two eq '${') {
        my $nj = sh_skip_balanced($line, $p, '${', $mdq);
        if (defined $nj) { $p = $nj; next }
        $p += 2;
        next;
      }
      if (!$msq && $c eq '`') {
        my $nj = sh_skip_balanced($line, $p, '`');
        if (defined $nj) { $p = $nj; next }
        $in_bt = 1;
        $unfilt[$i] = 1;
        $p++;
        next;
      }
      if (@sub_stk && !$msq && !$mdq) {
        if ($c eq "(") { push @sub_stk, ["paren", 0]; $p++; next }
        if ($c eq ")") {
          if ($sub_stk[-1][0] eq "sub") { $mdq = (pop @sub_stk)->[1] }
          elsif ($sub_stk[-1][0] eq "paren") { pop @sub_stk }
          $p++;
          next;
        }
        if (($p == 0 || substr($line, $p - 1, 1) =~ /[\s;|&(]/) && substr($line, $p) =~ /\A(case|esac)(?=[\s;|&()]|\z)/) {
          my $kw = $1;
          if ($kw eq "case") { push @sub_stk, ["case", 0] }
          elsif ($sub_stk[-1][0] eq "case") { pop @sub_stk }
          $p += length($kw);
          next;
        }
      }
      $p++;
    }
    $i++;
  }
  return $sh_base{$f} = {
    lines => \@l, kind => \@kind, closed_hds => \@closed_hds,
    unfilt => \@unfilt, in_fn => \@in_fn, funcs => \%funcs, line_toks => \%line_toks,
  };
}
# Classify a `.sh` file's heredocs and build its filtered scan lines.
# Cached per ($f, overridden-command key); most files share the empty-override entry.
sub sh_parse {
  my ($f, $overridden) = @_;
  my $okey = $overridden ? join(",", sort grep { $overridden->{$_} } keys %$overridden) : "";
  my $ckey = "$f\0$okey";
  return $sh_parsed{$ckey} if $sh_parsed{$ckey};
  my $base = sh_parse_base($f);
  my @l = @{$base->{lines}};
  my $ov_hd = ($overridden && ($overridden->{cat} || $overridden->{tee}
    || $overridden->{"\0exec_stdout"} || $overridden->{"\0exec_stderr"} || $overridden->{"\0exec_any_fd"})) ? 1 : 0;
  my ($kinds_ref, $funcs_ref, $unq_ref);
  my $unfilt_ref = $base->{unfilt};
  my $infn_ref   = $base->{in_fn};
  if (!@{$base->{closed_hds}}) {
    $kinds_ref = $base->{kind};
    $funcs_ref = $base->{funcs};
    $unq_ref   = [];
  } elsif (!$ov_hd && $sh_parsed{"$f\0"}) {
    $kinds_ref = $sh_parsed{"$f\0"}{kinds};
    $funcs_ref = $sh_parsed{"$f\0"}{funcs};
    $unq_ref   = $sh_parsed{"$f\0"}{unq};
  } else {
    my @kind = @{$base->{kind}};
    my %funcs = %{$base->{funcs}};
    my @closed_hds = @{$base->{closed_hds}};
    my @unfilt = @$unfilt_ref;
    my @in_fn  = @$infn_ref;
    my @unq_spans;
    # Pass 1c: classify every closed heredoc (and scan executable heredocs for
    # command overrides, iterating to a fixpoint if `hd_exec` defines `cat`/`tee`).
    my %eff_ov = (%funcs, $overridden ? %$overridden : ());
    my $hd_changed = 1;
    while ($hd_changed) {
      $hd_changed = 0;
      @unq_spans = ();
      for my $hd (@closed_hds) {
        my $op_unfilt = (grep { $unfilt[$_] } $hd->{opener} .. $hd->{opener_end}) ? 1 : 0;
        my $op_in_fn  = (grep { $in_fn[$_] }  $hd->{opener} .. $hd->{opener_end}) ? 1 : 0;
        my $is_data = !$op_unfilt && heredoc_is_data($hd->{op_text}, $hd->{op_pos}, \%eff_ov, $op_in_fn);
        if ($is_data) {
          my $bk = $hd->{quoted} ? "hd_quoted" : "hd_unquoted";
          $kind[$_] = $bk for @{$hd->{body}};
          push @unq_spans, [@{$hd->{body}}] if !$hd->{quoted} && @{$hd->{body}};
        } else {
          $kind[$_] = "hd_exec" for @{$hd->{body}};
          for my $bi (@{$hd->{body}}) {
            my %bf;
            sh_scan_overrides($l[$bi], \%bf);
            for my $fn (keys %bf) {
              $funcs{$fn} = 1;
              if (!$eff_ov{$fn}) {
                $eff_ov{$fn} = 1;
                $hd_changed = 1 if $fn eq "cat" || $fn eq "tee"
                  || $fn eq "\0exec_stdout" || $fn eq "\0exec_stderr" || $fn eq "\0exec_any_fd";
              }
            }
          }
        }
      }
    }
    $kinds_ref = \@kind;
    $funcs_ref = \%funcs;
    $unq_ref   = \@unq_spans;
  }
  my @kind = @$kinds_ref;
  my @scan = @l;
  # Extract command substitutions across each whole unquoted data heredoc body
  # so multiline $(...) / `...` stay intact with exact physical line numbers.
  for my $span (@$unq_ref) {
    my $whole = join "\n", map { $l[$_] } @$span;
    my @ex = split /\n/, extract_cmdsubs($whole, 0), -1;
    for my $k (0 .. $#$span) { $scan[$span->[$k]] = $ex[$k] // "" }
  }
  my %eff_scan = (%$funcs_ref, $overridden ? %$overridden : ());
  my $i = 0;
  while ($i <= $#l) {
    if ($kind[$i] eq "hd_quoted" || $kind[$i] eq "hd_delim") { $scan[$i] = ""; $i++; next }
    if ($kind[$i] ne "code" || $unfilt_ref->[$i]) { $i++; next }
    if ($l[$i] =~ /^\s*#/) { $i++; next }
    if (exists $base->{line_toks}{$i}) {
      if (my $lt = $base->{line_toks}{$i}) {
        $scan[$i] = filter_sh_logical($l[$i], \%eff_scan, $infn_ref->[$i] ? 1 : 0, $lt);
      }
      $i++;
      next;
    }
    # Gather a logical shell line across trailing `\` or an open multiline quote
    # in O(line length), failing closed if quotes never close, hit an unfiltered
    # region, or split a substitution opener across `\`-newline (`$\` + `(`).
    my $j = $i;
    my @parts = ($l[$i]);
    my ($lsq, $ldq, $split_sub) = (0, 0, 0);
    my $scan_q = sub {
      my ($str) = @_;
      my ($esc, $p, $len) = (0, 0, length $str);
      while ($p < $len) {
        my $c = substr($str, $p, 1);
        if ($esc) { $esc = 0; $p++; next }
        if ($c eq "\\" && !$lsq) { $esc = 1; $p++; next }
        if ($c eq "'" && !$ldq) { $lsq = !$lsq; $p++; next }
        if ($c eq '"' && !$lsq) { $ldq = !$ldq; $p++; next }
        if (!$lsq && !$ldq && $c eq "#" && ($p == 0 || substr($str, $p - 1, 1) =~ /[\s;|&(]/)) { last }
        $p++;
      }
      return $esc;
    };
    my $trail_esc = $scan_q->($parts[0]);
    while ($j < $#l && $kind[$j + 1] eq "code" && !$unfilt_ref->[$j + 1] && ($lsq || $ldq || $trail_esc)) {
      if ($trail_esc && !$lsq && !$ldq) {
        $split_sub = 1 if $parts[-1] =~ /[\$<>]\\\z/;
        $parts[-1] =~ s/\\\z/ /;
      }
      $j++;
      push @parts, $l[$j];
      $trail_esc = $scan_q->($parts[-1]);
    }
    if (!$lsq && !$ldq && !$split_sub) {
      my $joined = join "\n", @parts;
      my $blk_fn = (grep { $infn_ref->[$_] } $i .. $j) ? 1 : 0;
      my @fl = split /\n/, filter_sh_logical($joined, \%eff_scan, $blk_fn), -1;
      for my $k (0 .. $#fl) { $scan[$i + $k] = $fl[$k] }
      $i = $j + 1;
    } else {
      # Unclosed quote or split substitution opener: leave lines raw (fail closed).
      $i = $split_sub ? $j + 1 : $i + 1;
    }
  }
  return $sh_parsed{$ckey} = { kinds => $kinds_ref, scan => \@scan, funcs => $funcs_ref, unq => $unq_ref, unfilt => $unfilt_ref, in_fn => $infn_ref };
}
sub assigns {   # (FILE [, OVERRIDES]) -> {V => how many writes of V it may hold}
  my ($f, $ov) = @_;
  my $okey = $ov ? join(",", sort grep { $ov->{$_} } keys %$ov) : "";
  my $ckey = "$f\0$okey";
  return $assigns{$ckey} if $assigns{$ckey};
  my %n;
  my @l = split /\n/, raw($f);
  my $sh_p = $f =~ /\.sh\z/ ? sh_parse($f, $ov) : undef;
  for my $i (0 .. $#l) {
    my $k = $sh_p ? $sh_p->{kinds}[$i] : "code";
    next if $k eq "hd_quoted" || $k eq "hd_delim";
    my $line = $k eq "hd_unquoted" ? $sh_p->{scan}[$i] : $l[$i];
    next if $k ne "hd_unquoted" && $line =~ /^\s*#/;
    $n{$_}++ for $line =~ /(?:\A|[\s;&|(`\{])(?:(?:local|export|readonly|declare|typeset)(?:\s+-\w+)*\s+)?([A-Za-z_]\w*)(?:\+|:)?=/g;
    $n{$_}++ for $line =~ /\bfor\s+([A-Za-z_]\w*)\s/g;
    if ($line =~ /\b(?:read|readarray|mapfile|getopts|unset|printf\s+-v)\b/) { my %v = map { $_ => 1 } $line =~ /([A-Za-z_]\w*)/g; $n{$_}++ for keys %v }
  }
  return $assigns{$ckey} = \%n;
}
# FILE -> {V => [`0` or `S`, UPS]} for each V its one assignment sets to the
# self-directory form, UPS `/..` above it.
my $self_dir = qr{\$\((?:CDPATH=\s*)?cd\s+(?:--\s+)?"\$\(dirname\s+(?:--\s+)?($self_arg)\)((?:/\.\.)*)"(?:\s*2>\s*/dev/null)?\s*&&\s*pwd(?:\s+-P)?\)};
sub selfdirs {
  my ($f, $ov) = @_;
  my $okey = $ov ? join(",", sort grep { $ov->{$_} } keys %$ov) : "";
  my $ckey = "$f\0$okey";
  return $selfdirs{$ckey} if $selfdirs{$ckey};
  my %r;
  if ($f =~ /\.sh\z/) {
    my @l = split /\n/, raw($f);
    my $kinds = sh_parse($f, $ov)->{kinds};
    for my $i (0 .. $#l) {
      next if $kinds->[$i] ne "code";
      my $line = $l[$i];
      next if $line =~ /^\s*#/;
      next unless $line =~ /\A\s*(?:(?:local|export|readonly)\s+)?([A-Za-z_]\w*)=("?)$self_dir\2\s*(?:\|\|.*)?\z/;
      my ($v, $arg, $ups) = ($1, $3, length($4) / 3);
      $r{$v} = [$arg =~ /BASH_SOURCE/ ? "S" : "0", $ups] if assigns($f, $ov)->{$v} == 1;
    }
  }
  return $selfdirs{$ckey} = \%r;
}
# Python __file__-relative path helpers (gh-ocannl-2380).
# Single-pass scan of a `.py` source: returns
#   ($whole_code_no_strings, \@no_comment_lines, \@in_triple, \@fstr_exprs)
# where $whole_code_no_strings blanks strings and `#` comments (preserving `\n`),
# @no_comment_lines strips only `#` comments (keeping strings intact),
# $in_triple->[$i] is 1 when physical line $i begins inside a triple-quoted string,
# and @fstr_exprs holds the raw `{...}` replacement fields extracted from f-strings
# so walrus bindings like `f'{(HERE := "data")}'` are never hidden.
sub py_strip_file {
  my ($s) = @_;
  my ($out, $nc, $i, $len, $ln) = ("", "", 0, length($s), 0);
  my ($q, $triple, $esc, $is_fstr) = ("", 0, 0, 0);
  my @in_triple = (0);
  my @fstr_exprs;
  while ($i < $len) {
    my $c = substr($s, $i, 1);
    if ($c eq "\n") {
      $out .= "\n";
      $nc  .= "\n";
      $esc = 0;
      if (!$triple) { $q = ""; $is_fstr = 0 }
      $ln++;
      $in_triple[$ln] = $triple ? 1 : 0;
      $i++;
      next;
    }
    if ($q ne "") {
      if ($esc) { $esc = 0; $out .= " "; $nc .= $c; $i++; next }
      if ($c eq "\\") { $esc = 1; $out .= " "; $nc .= $c; $i++; next }
      if ($is_fstr && $c eq "{") {
        if (substr($s, $i, 2) eq "{{") {
          $out .= "  "; $nc .= "{{"; $i += 2; next;
        }
        $out .= " "; $nc .= "{";
        my ($j, $bdepth, $fsq, $fdq, $fesc) = ($i + 1, 1, 0, 0, 0);
        while ($j < $len && $bdepth > 0) {
          my $fc = substr($s, $j, 1);
          if ($fc eq "\n") {
            $out .= "\n"; $nc .= "\n"; $fesc = 0; $ln++;
            $in_triple[$ln] = $triple ? 1 : 0;
            $j++; next;
          }
          $out .= " "; $nc .= $fc;
          if ($fesc) { $fesc = 0; $j++; next }
          if ($fc eq "\\" && ($fsq || $fdq)) { $fesc = 1; $j++; next }
          if ($fc eq "'" && !$fdq) { $fsq = !$fsq; $j++; next }
          if ($fc eq '"' && !$fsq) { $fdq = !$fdq; $j++; next }
          if (!$fsq && !$fdq) {
            if ($fc eq "{") { $bdepth++ }
            elsif ($fc eq "}") { $bdepth-- }
          }
          $j++;
        }
        push @fstr_exprs, substr($s, $i + 1, $j - $i - ($bdepth == 0 ? 2 : 1));
        $i = $j;
        next;
      }
      if ($triple && substr($s, $i, 3) eq ($q x 3)) {
        $out .= "   ";
        $nc  .= $q x 3;
        $q = ""; $triple = 0; $is_fstr = 0;
        $i += 3;
        next;
      }
      if (!$triple && $c eq $q) { $out .= " "; $nc .= $c; $q = ""; $is_fstr = 0; $i++; next }
      $out .= " ";
      $nc  .= $c;
      $i++;
      next;
    }
    if ($c eq "#") {
      my $e = index($s, "\n", $i);
      $e = $len if $e < 0;
      my $sp = " " x ($e - $i);
      $out .= $sp;
      $nc  .= $sp;
      $i = $e;
      next;
    }
    if ($c eq "'" || $c eq '"') {
      my $pre = substr($s, $i > 3 ? $i - 3 : 0, $i > 3 ? 3 : $i);
      $is_fstr = ($pre =~ /(?:^|[^\w])[rRfF]*[fF][rRfF]*\z/) ? 1 : 0;
      if (substr($s, $i, 3) eq ($c x 3)) {
        $q = $c; $triple = 1;
        $out .= "   ";
        $nc  .= $c x 3;
        $i += 3;
        next;
      }
      $q = $c; $triple = 0;
      $out .= " ";
      $nc  .= $c;
      $i++;
      next;
    }
    $out .= $c;
    $nc  .= $c;
    $i++;
  }
  return ($out, [split /\n/, $nc, -1], \@in_triple, \@fstr_exprs);
}
# Parse a Python self-directory expression -> [UPS, IS_REAL] or undef, where
# IS_REAL is 1 when `.resolve()` or `os.path.realpath` dereferences symlinks.
sub py_self_dir_ups {
  my ($expr, $rebound) = @_;
  return undef if $rebound->{__file__};
  $expr =~ s/\A\s+//; $expr =~ s/\s+\z//;
  if ($expr =~ /\Astr\s*\(\s*(.+?)\s*\)\z/) {
    return undef if $rebound->{str};
    $expr = $1;
  }
  # Nested os.path.dirname(...) around __file__, os.path.abspath(__file__), or os.path.realpath(__file__).
  if ($expr =~ /\Aos\.path\.dirname\s*\(/) {
    return undef if $rebound->{os};
    my ($cur, $depth) = ($expr, 0);
    while ($cur =~ /\Aos\.path\.dirname\s*\(\s*(.+?)\s*\)\z/) { $cur = $1; $depth++ }
    if ($depth >= 1) {
      return [$depth - 1, 0] if $cur =~ /\A(?:__file__|os\.path\.abspath\s*\(\s*__file__\s*\))\z/;
      return [$depth - 1, 1] if $cur =~ /\Aos\.path\.realpath\s*\(\s*__file__\s*\)\z/;
    }
    return undef;
  }
  # (pathlib.)?Path(__file__)(.resolve())?(.parent+ | .parents[N])
  if ($expr =~ /\A(pathlib\.)?Path\s*\(\s*__file__\s*\)(\s*\.\s*resolve\s*\(\s*\))?\s*((?:\.\s*parent\b)+|\.\s*parents\s*\[\s*(\d+)\s*\])\z/) {
    my ($mod, $res, $tail, $idx) = ($1, $2, $3, $4);
    return undef if defined $mod ? $rebound->{pathlib} : $rebound->{Path};
    my $is_real = defined($res) ? 1 : 0;
    return [$idx + 0, $is_real] if defined $idx;
    my @p = $tail =~ /\.\s*parent\b/g;
    return @p ? [scalar(@p) - 1, $is_real] : undef;
  }
  return undef;
}
# Resolve the repo-relative anchor directory for Python file $f given [UPS, IS_REAL].
# When IS_REAL is set and $f (or a parent directory) is a symlink, dereference to
# the symlink target's repo-relative path (returning undef to fail closed if the
# target lies outside the repository).
my $repo_real;
sub py_anchor_dir {
  my ($f, $spec) = @_;
  my ($ups, $is_real) = @$spec;
  my $base_f = $f;
  if ($is_real) {
    my $has_link = -l $f ? 1 : 0;
    if (!$has_link) {
      my @segs = split m{/}, $f;
      pop @segs;
      my $cur = "";
      for my $seg (@segs) {
        $cur = $cur eq "" ? $seg : "$cur/$seg";
        if (-l $cur) { $has_link = 1; last }
      }
    }
    if ($has_link) {
      $repo_real //= Cwd::realpath(".");
      my $rf = Cwd::realpath($f);
      return undef unless defined $repo_real && defined $rf;
      my $prefix = $repo_real eq "/" ? "/" : "$repo_real/";
      return undef unless index($rf, $prefix) == 0;
      $base_f = substr($rf, length($prefix));
    }
  }
  return norm(join "/", dir_of($base_f), ("..") x $ups);
}
# Analyze a `.py` file once: { sd => {V => UPS}, rebound => {...}, no_comment => \@, in_triple => \@ }.
sub py_analyze {
  my ($f) = @_;
  return $py_info{$f} if $py_info{$f};
  my (%writes, %rebound, %sd);
  my %special = map { $_ => 1 } qw(os pathlib Path __file__ str);
  my %py_non_cap = map { $_ => 1 } qw(_ True False None as if and or not in is);
  my $note_bind = sub {
    my ($v) = @_;
    $writes{$v}++;
    $rebound{$v} = 1 if $special{$v};
  };
  my $src = raw($f);
  my ($whole_code, $no_comment, $in_triple, $fstr_exprs) = py_strip_file($src);
  for my $fe (@$fstr_exprs) {
    $note_bind->($_) for $fe =~ /([A-Za-z_]\w*)\s*:=/g;
  }
  my @l = split /\n/, $src, -1;
  # Join physical lines into logical Python lines across open (), [], {} or trailing `\`.
  my (@logical, $buf, $bdepth);
  ($buf, $bdepth) = ("", 0);
  for my $raw_cl (split /\n/, $whole_code, -1) {
    my $cont = ($raw_cl =~ /\\\s*\z/) ? 1 : 0;
    my $piece = $raw_cl;
    $piece =~ s/\\\s*\z/ / if $cont;
    $buf .= ($buf eq "" ? "" : " ") . $piece;
    $bdepth += () = $piece =~ /[\(\[\{]/g;
    $bdepth -= () = $piece =~ /[\)\]\}]/g;
    $bdepth = 0 if $bdepth < 0;
    if ($bdepth == 0 && !$cont) {
      push @logical, $buf;
      $buf = "";
    }
  }
  push @logical, $buf if $buf =~ /\S/;
  for my $code (@logical) {
    next unless $code =~ /\S/;
    $rebound{os}      = 1 if $code =~ /\bos\.path(?:\.\w+)?\s*=(?!=)/;
    $rebound{pathlib} = 1 if $code =~ /\bpathlib\.Path\s*=(?!=)/;
    $rebound{Path}    = 1 if $code =~ /\bPath\.\w+\s*=(?!=)/;
    # Split each logical line on `;` before handling imports and assignment targets
    # (so `import sys; HERE = 'data'` and `if flag: import fake as os; HERE = ...`
    # analyze every statement).
    for my $seg (split /;/, $code) {
      my $stmt = $seg;
      if ($stmt =~ /\A\s*(?:async\s+)?(if|elif|else|for|while|with|try|except|finally|match|case|def|class)\b/) {
        my $kw = $1;
        my ($p, $depth, $slen, $colon) = (0, 0, length($stmt), -1);
        while ($p < $slen) {
          my $c = substr($stmt, $p, 1);
          if ($c =~ /[\(\[\{]/) { $depth++ }
          elsif ($c =~ /[\)\]\}]/) { $depth-- }
          elsif ($c eq ":" && $depth == 0 && substr($stmt, $p, 2) ne ":=") { $colon = $p; last }
          $p++;
        }
        if ($kw eq "case") {
          my $pat = $colon >= 0 ? substr($stmt, 0, $colon) : $stmt;
          $pat =~ s/\A\s*case\b//;
          $pat =~ s/\bif\b.*//;
          for my $id ($pat =~ /\b([A-Za-z_]\w*)\b/g) {
            $note_bind->($id) unless $py_non_cap{$id};
          }
        }
        $stmt = $colon >= 0 ? substr($stmt, $colon + 1) : "";
      }
      # Track non-standard bindings of os, pathlib, Path, __file__, str.
      if ($stmt =~ /\A\s*import\s+(.+)/) {
        for my $part (split /,/, $1) {
          if ($part =~ /\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+as\s+([A-Za-z_]\w*)/) {
            $note_bind->($2);
          } elsif ($part =~ /\A\s*([A-Za-z_]\w*)/) {
            $note_bind->($1) unless $1 eq "os" || $1 eq "pathlib";
          }
        }
        next;
      }
      if ($stmt =~ /\A\s*from\s+(\S+)\s+import\s+(.+)/) {
        my ($mod, $rest) = ($1, $2);
        if ($rest =~ /\*/) {
          $rebound{$_} = 1 for keys %special;
          next;
        }
        $rest =~ s/[()]//g;
        for my $part (split /,/, $rest) {
          if ($part =~ /\b([A-Za-z_]\w*)\s+as\s+([A-Za-z_]\w*)/) {
            $note_bind->($2);
          } elsif ($part =~ /\A\s*([A-Za-z_]\w*)/) {
            $note_bind->($1) unless $mod eq "pathlib" && $1 eq "Path";
          }
        }
        next;
      }
      # Split $stmt on bracket-depth-0 binding `=` (not `==`, `!=`, `<=`, `>=`, `:=`).
      my @lhs_parts;
      my ($p, $depth, $slen, $start) = (0, 0, length($stmt), 0);
      while ($p < $slen) {
        my $c = substr($stmt, $p, 1);
        if ($c =~ /[\(\[\{]/) { $depth++ }
        elsif ($c =~ /[\)\]\}]/) { $depth-- }
        elsif ($c eq "=" && $depth == 0) {
          my $prev = $p > 0 ? substr($stmt, $p - 1, 1) : "";
          my $next = $p + 1 < $slen ? substr($stmt, $p + 1, 1) : "";
          if ($next eq "=") { $p += 2; next }
          if ($prev eq "!" || $prev eq "<" || $prev eq ">" || $prev eq ":") { $p++; next }
          push @lhs_parts, substr($stmt, $start, $p - $start);
          $start = $p + 1;
        }
        $p++;
      }
      for my $lhs (@lhs_parts) {
        # Strip trailing augmented-assignment operator and top-level `: annotation`.
        $lhs =~ s/(?:\+|-|\*|\/|%|&|\||\^|<<|>>|\*\*|\/\/)\s*\z//;
        my ($lp, $ld, $llen, $lcolon) = (0, 0, length($lhs), -1);
        while ($lp < $llen) {
          my $lc = substr($lhs, $lp, 1);
          if ($lc =~ /[\(\[\{]/) { $ld++ }
          elsif ($lc =~ /[\)\]\}]/) { $ld-- }
          elsif ($lc eq ":" && $ld == 0) { $lcolon = $lp; last }
          $lp++;
        }
        $lhs = substr($lhs, 0, $lcolon) if $lcolon >= 0;
        next unless $lhs =~ /\A[\sA-Za-z0-9_,\(\)\[\]\*]+\z/;
        next if $lhs =~ /[A-Za-z_]\w*\s*[\(\[]/;
        $note_bind->($_) for $lhs =~ /\b([A-Za-z_]\w*)\b/g;
      }
    }
    $note_bind->($_) for $code =~ /([A-Za-z_]\w*)\s*:=/g;
    for my $for_vars ($code =~ /\bfor\s+([\sA-Za-z0-9_,\(\)\[\]\*]+?)\s+in\b/g) {
      next if $for_vars =~ /[A-Za-z_]\w*\s*[\(\[]/;
      $note_bind->($_) for $for_vars =~ /\b([A-Za-z_]\w*)\b/g;
    }
    for my $as_vars ($code =~ /\bas\s+([\sA-Za-z0-9_,\(\)\[\]\*]+)/g) {
      next if $as_vars =~ /[A-Za-z_]\w*\s*[\(\[]/;
      for my $id ($as_vars =~ /\b([A-Za-z_]\w*)\b/g) { $note_bind->($id) unless $py_non_cap{$id} }
    }
    $note_bind->($_) for $code =~ /\b(?:def|class)\s+([A-Za-z_]\w*)/g;
    for my $del_vars ($code =~ /\bdel\s+([\sA-Za-z0-9_,\(\)\[\]]+)/g) {
      next if $del_vars =~ /[A-Za-z_]\w*\s*[\(\[]/;
      $note_bind->($_) for $del_vars =~ /\b([A-Za-z_]\w*)\b/g;
    }
  }
  # Multiline-safe function and lambda parameter scanning over $whole_code
  # (handles multiline def signatures, nested default parens, and type annotations).
  while ($whole_code =~ /\bdef\s+[A-Za-z_]\w*\s*\(/g) {
    my ($p, $depth, $wlen) = (pos($whole_code), 1, length($whole_code));
    my $start = $p;
    while ($p < $wlen && $depth > 0) {
      my $c = substr($whole_code, $p, 1);
      if ($c eq "(") { $depth++ } elsif ($c eq ")") { $depth-- }
      $p++;
    }
    next unless $depth == 0;
    my $plist = substr($whole_code, $start, $p - $start - 1);
    for my $param (py_split_args($plist)) {
      $param =~ s/=.*//s; $param =~ s/:.*//s;
      $note_bind->($1) if $param =~ /\b([A-Za-z_]\w*)\b/;
    }
  }
  while ($whole_code =~ /\blambda\b/g) {
    my ($p, $depth, $wlen) = (pos($whole_code), 0, length($whole_code));
    my $start = $p;
    my $end = -1;
    while ($p < $wlen) {
      my $c = substr($whole_code, $p, 1);
      if ($c =~ /[\(\[\{]/) { $depth++ }
      elsif ($c =~ /[\)\]\}]/) { $depth-- }
      elsif ($c eq ":" && $depth == 0) { $end = $p; last }
      $p++;
    }
    next if $end < 0;
    my $plist = substr($whole_code, $start, $end - $start);
    for my $param (py_split_args($plist)) {
      $param =~ s/=.*//s;
      $note_bind->($1) if $param =~ /\b([A-Za-z_]\w*)\b/;
    }
  }
  for my $i (0 .. $#l) {
    next if $in_triple->[$i] || $l[$i] =~ /^\s*#/;
    my $line = $no_comment->[$i];
    next unless $line =~ /\A\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(.+?)\s*\z/;
    my ($v, $rhs) = ($1, $2);
    next unless ($writes{$v} // 0) == 1;
    my $ups = py_self_dir_ups($rhs, \%rebound);
    $sd{$v} = $ups if defined $ups;
  }
  return $py_info{$f} = { sd => \%sd, rebound => \%rebound, no_comment => $no_comment, in_triple => $in_triple };
}
# Split top-level comma-separated Python arguments, respecting quotes & parens.
sub py_split_args {
  my ($s) = @_;
  my (@args, $cur);
  $cur = "";
  my ($sq, $dq, $esc, $depth, $i, $len) = (0, 0, 0, 0, 0, length $s);
  while ($i < $len) {
    my $c = substr($s, $i, 1);
    if ($esc) { $cur .= $c; $esc = 0; $i++; next }
    if ($c eq "\\" && ($sq || $dq)) { $cur .= $c; $esc = 1; $i++; next }
    if ($c eq "'" && !$dq) { $sq = !$sq; $cur .= $c; $i++; next }
    if ($c eq '"' && !$sq) { $dq = !$dq; $cur .= $c; $i++; next }
    if (!$sq && !$dq) {
      if ($c =~ /[\(\[\{]/) { $depth++ }
      elsif ($c =~ /[\)\]\}]/) { $depth-- }
      elsif ($c eq "," && $depth == 0) { push @args, $cur; $cur = ""; $i++; next }
    }
    $cur .= $c;
    $i++;
  }
  push @args, $cur if $cur =~ /\S/;
  return @args;
}
# Extract placed __file__-relative path joins from a comment-stripped `.py` line.
# Returns ($remaining_line, @placed) where each placed entry is [$rel, $at].
sub py_placed_runs {
  my ($line, $f) = @_;
  my $info = py_analyze($f);
  my ($sd, $rebound) = ($info->{sd}, $info->{rebound});
  my $rem = $line;
  my @placed;
  my ($sq, $dq, $esc, $i, $len) = (0, 0, 0, 0, length $line);
  while ($i < $len) {
    my $c = substr($line, $i, 1);
    if ($esc) { $esc = 0; $i++; next }
    if ($c eq "\\" && ($sq || $dq)) { $esc = 1; $i++; next }
    if ($c eq "'" && !$dq) { $sq = !$sq; $i++; next }
    if ($c eq '"' && !$sq) { $dq = !$dq; $i++; next }
    if ($sq || $dq) { $i++; next }
    my $pre_ok = ($i == 0 || substr($line, $i - 1, 1) !~ /[.\w]/);
    my $rest = substr($line, $i);
    # Form 1: os.path.join(BASE, LIT1 [, LIT2 ...])
    if ($pre_ok && !$rebound->{os} && $rest =~ /\A(os\.path\.join\s*\()/) {
      my $open_end = $i + length($1);
      my ($j, $depth, $isq, $idq, $iesc) = ($open_end, 1, 0, 0, 0);
      while ($j < $len && $depth > 0) {
        my $cc = substr($line, $j, 1);
        if ($iesc) { $iesc = 0; $j++; next }
        if ($cc eq "\\" && ($isq || $idq)) { $iesc = 1; $j++; next }
        if ($cc eq "'" && !$idq) { $isq = !$isq; $j++; next }
        if ($cc eq '"' && !$isq) { $idq = !$idq; $j++; next }
        if (!$isq && !$idq) {
          if ($cc eq "(") { $depth++ } elsif ($cc eq ")") { $depth-- }
        }
        $j++;
      }
      if ($depth == 0) {
        my @args = py_split_args(substr($line, $open_end, $j - $open_end - 1));
        if (@args >= 2) {
          my $base = shift @args;
          $base =~ s/\A\s+//; $base =~ s/\s+\z//;
          my $ups = py_self_dir_ups($base, $rebound);
          $ups = $sd->{$base} if !defined($ups) && $base =~ /\A[A-Za-z_]\w*\z/ && exists $sd->{$base};
          if (defined $ups) {
            my @parts;
            my $all_lit = 1;
            for my $a (@args) {
              $a =~ s/\A\s+//; $a =~ s/\s+\z//;
              if ($a =~ m{\A(['"])([A-Za-z0-9_.-][A-Za-z0-9_./-]*)\1\z}) { push @parts, $2 }
              else { $all_lit = 0; last }
            }
            if ($all_lit && @parts) {
              my $at = py_anchor_dir($f, $ups);
              if (defined $at) {
                push @placed, [join("/", @parts), $at];
                substr($rem, $i, $j - $i) = " " x ($j - $i);
                $i = $j;
                next;
              }
            }
          }
        }
      }
    }
    # Form 2: BASE / LIT1 [/ LIT2 ...] (not preceded by `/` or `.`, not followed by `/`).
    my $slash_pre_ok = ($i == 0 || substr($line, $i - 1, 1) !~ /[\/.\w]/);
    if ($slash_pre_ok) {
      my ($base_len, $ups);
      if ($rest =~ /\A((?:pathlib\.)?Path\s*\(\s*__file__\s*\)(?:\s*\.\s*resolve\s*\(\s*\))?\s*(?:(?:\.\s*parent\b)+|\.\s*parents\s*\[\s*\d+\s*\]))/) {
        $base_len = length($1);
        $ups = py_self_dir_ups($1, $rebound);
      } elsif ($rest =~ /\A([A-Za-z_]\w*)(?!\w)/ && exists $sd->{$1}) {
        $base_len = length($1);
        $ups = $sd->{$1};
      }
      if (defined $ups) {
        my $after = substr($rest, $base_len);
        my ($consumed, @parts) = (0);
        while ($after =~ m{\G(\s*/\s*(['"])([A-Za-z0-9_.-][A-Za-z0-9_./-]*)\2)}gc) {
          push @parts, $3;
          $consumed = pos($after);
        }
        if (@parts && substr($after, $consumed) !~ /\A\s*\//) {
          my $at = py_anchor_dir($f, $ups);
          if (defined $at) {
            my $span = $base_len + $consumed;
            push @placed, [join("/", @parts), $at];
            substr($rem, $i, $span) = " " x $span;
            $i += $span;
            next;
          }
        }
      }
    }
    $i++;
  }
  return ($rem, @placed);
}
# FILE in harness H's program PROG -> {PATH => [LINE, RUN, OWNERS]}: the
# first run naming each path, and how many paths that run named.
sub names_of {
  my ($f, $h, $prog) = @_;
  return $names{"$f\0$h"} if $names{"$f\0$h"};
  my ($sh_p, $sh_scan, %ov);
  if ($f =~ /\.sh\z/) {
    for my $pf (@$prog) {
      next unless $pf =~ /\.sh\z/;
      my $pf_funcs = sh_parse($pf)->{funcs};
      $ov{$_} = 1 for keys %$pf_funcs;
    }
    # A sourced helper's stdout/stderr may be captured or redirected by its
    # caller (`code=$(source helper.sh)`, `source helper.sh | bash`, ...), so
    # require explicit plain-file stdout redirection just as in function bodies.
    if ($f ne $h) {
      $ov{"\0exec_stdout"} = 1;
      $ov{"\0exec_stderr"} = 1;
      $ov{"\0exec_any_fd"} = 1;
    }
    $sh_p = sh_parse($f, \%ov);
    $sh_scan = $sh_p->{scan};
  }
  my $sd = selfdirs($f, \%ov);
  my $py_p = $f =~ /\.py\z/ ? py_analyze($f) : undef;
  my %r;
  my @l = split /\n/, raw($f);
  for my $i (0 .. $#l) {
    if ($sh_p && $sh_p->{kinds}[$i] eq "hd_unquoted") {
      # Unquoted heredoc lines may begin with `#` and still expand $(...) / `...`.
      for my $run (runs($sh_scan->[$i])) {
        my ($t, $frag, $a) = @$run;
        my $at;
        if (defined $a) {
          my $form = $a eq '$0' ? ["0", 0] : $a eq '$BASH_SOURCE' ? ["S", 0]
            : $sd->{$a} && !(grep { $_ ne $f && assigns($_, \%ov)->{$a} } @$prog) ? $sd->{$a} : undef;
          $at = norm(join "/", dir_of($form->[0] eq "S" ? $f : $h), ("..") x $form->[1]) if $form;
        }
        my @p = resolve($t, $frag, $at);
        $r{$_} //= [$i + 1, $t, scalar @p] for @p;
      }
      next;
    }
    next if $l[$i] =~ /^\s*#/;
    my $line = $sh_scan ? $sh_scan->[$i] : $py_p ? $py_p->{no_comment}[$i] : $l[$i];
    if ($py_p && !$py_p->{in_triple}[$i]) {
      my ($rem, @placed) = py_placed_runs($line, $f);
      for my $pl (@placed) {
        my ($t, $at) = @$pl;
        my @p = resolve($t, 0, $at);
        $r{$_} //= [$i + 1, $t, scalar @p] for @p;
      }
      $line = $rem;
    }
    for my $run (runs($line)) {
      my ($t, $frag, $a) = @$run;
      my $at;
      if (defined $a && $f =~ /\.sh\z/) {
        my $form = $a eq '$0' ? ["0", 0] : $a eq '$BASH_SOURCE' ? ["S", 0]
          : $sd->{$a} && !(grep { $_ ne $f && assigns($_, \%ov)->{$a} } @$prog) ? $sd->{$a} : undef;
        $at = norm(join "/", dir_of($form->[0] eq "S" ? $f : $h), ("..") x $form->[1]) if $form;
      }
      my @p = resolve($t, $frag, $at);
      $r{$_} //= [$i + 1, $t, scalar @p] for @p;
    }
  }
  return $names{"$f\0$h"} = \%r;
}
sub loads_of {   # (FILE [, OVERRIDES]) -> {SCRIPT => [LINE, RUN, OWNERS]}, as names_of, for what it loads
  my ($f, $ov) = @_;
  my $okey = $ov ? join(",", sort grep { $ov->{$_} } keys %$ov) : "";
  my $ckey = "$f\0$okey";
  return $loads{$ckey} if $loads{$ckey};
  my %r;
  my @l = split /\n/, raw($f);
  my $sh_p;
  if ($f =~ /\.sh\z/) {
    my $base = sh_parse($f);
    my %eff = (%{$base->{funcs}}, $ov ? %$ov : ());
    $sh_p = %eff ? sh_parse($f, \%eff) : $base;
  }
  for my $i (0 .. $#l) {
    my $line = $l[$i];
    my @words;
    if ($f =~ /\.sh\z/) {
      my $k = $sh_p->{kinds}[$i];
      next if $k eq "hd_quoted" || $k eq "hd_delim";
      if ($k eq "hd_unquoted") {
        # Follow `.` / `source` executed inside $(...) / `...` of an unquoted heredoc,
        # never literal body text or fixture `# shellcheck source=` lines.
        my $scan = $sh_p->{scan}[$i];
        push @words, $1 if $scan =~ /(?:^|[;&|({`]|\bthen|\belse|\bdo)\s*(?:\.|source)\s+(.*)/;
      } elsif ($line =~ /^\s*#\s*shellcheck\s.*\bsource=(\S+)/) {
        push @words, $1;
      } elsif ($line !~ /^\s*#/) {
        my $scan = $sh_p->{scan}[$i];
        # A `.` or `source` command: the rest of its line names what it loads.
        push @words, $1 if $scan =~ /(?:^|[;&|({`]|\bthen|\belse|\bdo)\s*(?:\.|source)\s+(.*)/;
      }
    } elsif ($line =~ /^\s*import\s+([\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)/) {
      push @words, map { (s/\s.*//sr =~ tr{.}{/}r) . ".py" } split /\s*,\s*/, $1;
    } elsif ($line =~ /^\s*from\s+\.*([\w.]*)\s+import\s+\(?\s*(\w+(?:\s+as\s+\w+)?(?:\s*,\s*\w+(?:\s+as\s+\w+)?)*)/) {
      my ($base, $ns) = ($1 =~ tr{.}{/}r, $2);
      push @words, "$base.py" if $base ne "";
      push @words, map { ($base ne "" ? "$base/" : "") . (s/\s.*//sr) . ".py" } split /\s*,\s*/, $ns;
    }
    for my $run (map { runs($_) } @words) {
      my @p = resolve($run->[0], $run->[1]);
      $r{$_} //= [$i + 1, $run->[0], scalar @p] for grep { is_script($_) } @p;
    }
  }
  return $loads{$ckey} = \%r;
}
# Expand a worklist of scripts by `loads_of` to a fixpoint, propagating shell
# command overrides (`cat`/`tee`/`echo`/`printf`) discovered in any loaded `.sh`
# file back across the whole worklist.
sub expand_loads {
  my ($list, $par) = @_;
  my %ov;
  my $changed = 1;
  while ($changed) {
    $changed = 0;
    for (my $i = 0; $i < @$list; $i++) {
      my $sf = $list->[$i];
      next unless is_script($sf);
      if ($sf =~ /\.sh\z/) {
        my $funcs = sh_parse($sf)->{funcs};
        for my $fn (keys %$funcs) {
          if (!$ov{$fn}) { $ov{$fn} = 1; $changed = 1 }
        }
      }
      my $ld_ov = $i > 0 ? { %ov, "\0exec_stdout" => 1, "\0exec_stderr" => 1, "\0exec_any_fd" => 1 } : \%ov;
      my $ld = loads_of($sf, $ld_ov);
      for my $l (sort keys %$ld) {
        next if exists $par->{$l};
        $par->{$l} = [$sf, "loads", @{$ld->{$l}}];
        push @$list, $l;
        $changed = 1;
      }
    }
  }
}
# Each harness's reach, with the step that reached each path: its program (it
# and what it loads), what that names, and the program of each script named.
# A step is [FROM, `loads` or `names`, LINE, RUN, OWNERS].
my %reach;
for my $h (@members) {
  my %par = ($h => undef);
  my @prog = ($h);
  expand_loads(\@prog, \%par);
  my @named;
  for my $x (@prog) {
    my $nm = names_of($x, $h, \@prog);
    for my $z (sort keys %$nm) { next if exists $par{$z}; $par{$z} = [$x, "names", @{$nm->{$z}}]; push @named, $z }
  }
  my %named_ov;
  my $n_changed = 1;
  while ($n_changed) {
    $n_changed = 0;
    for (my $i = 0; $i < @named; $i++) {
      next unless is_script($named[$i]);
      if ($named[$i] =~ /\.sh\z/) {
        my $funcs = sh_parse($named[$i])->{funcs};
        for my $fn (keys %$funcs) {
          if (!$named_ov{$fn}) { $named_ov{$fn} = 1; $n_changed = 1 }
        }
      }
      my $ld = loads_of($named[$i], { %named_ov, "\0exec_stdout" => 1, "\0exec_stderr" => 1, "\0exec_any_fd" => 1 });
      for my $l (sort keys %$ld) { next if exists $par{$l}; $par{$l} = [$named[$i], "loads", @{$ld->{$l}}]; push @named, $l; $n_changed = 1 }
    }
  }
  $reach{$h} = \%par;
}
sub chain {   # how harness H reached P, read from its steps
  my ($h, $p) = @_;
  return "it is the harness" if $p eq $h;
  my @steps;
  for (my $q = $p; defined $reach{$h}{$q}; $q = $reach{$h}{$q}[0]) { unshift @steps, "$reach{$h}{$q}[1] $q" }
  return "it " . join(", which ", @steps);
}
# The declarations, from each harness's header (lines 2.. up to the first not
# led by `#`, as `--help` prints it); an `affected-reads` line anywhere else,
# or of any other shape, is refused.
my (%decl, @eres, @err);
for my $h (@members) {
  my @l = split /\n/, raw($h);
  my $header = 1;
  for my $i (0 .. $#l) {
    $header = 0 if $i >= 1 && $l[$i] !~ /^#/;
    next unless $l[$i] =~ /^\s*#\s*affected-reads\b/;
    my $where = "$h:" . ($i + 1);
    fail("$where: an affected-reads declaration outside the header (the comment block from line 2) is not read; move it there")
      if $i == 0 || !$header;
    $l[$i] =~ m{\A# affected-reads: (\S+(?: \S+)*?)(?: --only '([^'\t\n\r]+)')?\z}
      or fail("$where: not of the form `# affected-reads: GLOB... [--only 'ERE']`: $l[$i]");
    my ($globs, $ere) = ($1, $2);
    my @re;
    for my $g (split / /, $globs) {
      fail("$where: '$g' contains unsupported `**`; use `*` within one segment and an explicit list of directories for depth")
        if index($g, "**") >= 0;
      $g =~ m{\A[A-Za-z0-9_.*][A-Za-z0-9_.*-]*(?:/[A-Za-z0-9_.*-]+)*\z} && $g !~ m{(?:\A|/)\.\.?(?:/|\z)}
        or fail("$where: '$g' is not a repository path glob (letters, digits, `_.-*`, `/` between parts, no `.` or `..` part)");
      my $re = join "[^/]*", map { quotemeta } split /\*/, $g, -1;
      push @re, qr/\A$re\z/;
      push @err, "affected: warning: $where: $g matches no file here" unless grep { $_ =~ $re[-1] } @known;
    }
    push @{$decl{$h}}, { re => \@re, globs => [split / /, $globs], ere => $ere, line => $i + 1 };
    push @eres, "$ere\t$where" if defined $ere;
  }
}
# A marker grants no credit until this read proves it has no named consumer.
my %hand_run;
my $marker = "harness:";
my $sentinel = qr/#[ \t]*harness[ \t]*:/;
for my $p (grep { /\.sh\z/ && -f } @known) {
  my @lines = split /\r?\n/, raw($p);
  my $in_header = 1;
  for my $i (0 .. $#lines) {
    my $line = $lines[$i];
    $in_header = 0 if $i && $line !~ /^#/;
    my $markers = () = $line =~ /$sentinel/g;
    next unless $markers;
    my $where = "$p:" . ($i + 1);
    fail("$where: harness marker must follow the shebang") unless $i;
    fail("$where: harness marker must be in the header of a regular shell script")
      unless $in_header && $lines[0] =~ /^#!/ && !-l $p;
    fail("$where: harness marker requires a shell script path made of letters, digits, underscore, period, hyphen and slash")
      unless $p =~ m{\A[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.sh\z};
    fail("$where: duplicate harness marker") if $markers > 1 || $hand_run{$p};
    fail("$where: expected `# $marker none -- REASON` with a reason of at least two words")
      unless $line =~ /\A# \Q$marker\E none -- [!-~]+(?: [!-~]+)+\z/;
    $hand_run{$p} = $where;
  }
}
for my $p (sort keys %hand_run) {
  for my $h (@members) {
    next unless exists $reach{$h}{$p} || grep {
      my $d = $_; grep { $p =~ $_ } @{$d->{re}}
    } @{$decl{$h} // []};
    fail("$hand_run{$p}: harness marker conflicts with reach from $h");
  }
}
# Rule names and arguments are atoms, even within a quoted shell command.
# Compare literal basenames conservatively across directories and OS guards.
sub check_dune_markers {
  my ($file, $dir, $items, $by_base, $seen, $in_deps) = @_;
  for my $item (@$items) {
    if (ref $item) {
      my ($head, $arg) = @$item;
      $head = "" if !defined $head || ref $head;
      next if $head =~ /\A(?:glob_files|glob_files_rec|source_tree)\z/;
      if ($head eq "subdir" && defined $arg && !ref $arg && $arg !~ /%\{/) {
        my $sub = norm("$dir/$arg");
        check_dune_markers($file, $sub, [@$item[2 .. $#$item]], $by_base, $seen, 0) if defined $sub;
        next;
      }
      if (@$item == 2 && !ref $arg && $arg !~ m{(?:^/|%\{)}
          && ($head =~ /\A(?:include|dynamic_include)\z/ || ($in_deps && $head eq ":include"))) {
        # Stanza includes are relative to their source file. A dependency
        # include is a dependency of the enclosing stanza's directory.
        my $base = $head eq ":include" ? $dir : dir_of($file);
        my $path = norm("$base/$arg");
        read_dune_markers($path, $dir, $by_base, $seen, $head eq ":include") if defined $path && -f $path;
      }
      check_dune_markers($file, $dir, $item, $by_base, $seen, $in_deps || $head eq "deps");
      next;
    }
    while ($item =~ m{([A-Za-z0-9_.-]+\.sh)(?![A-Za-z0-9_.-])}g) {
      my $name = $1;
      my $p = $by_base->{$name} or next;
      fail("$hand_run{$p}: harness marker conflicts with a literal reference in $file");
    }
  }
}
sub read_dune_markers {
  my ($file, $dir, $by_base, $seen, $in_deps) = @_;
  return if $seen->{join("\0", $file, $dir, $in_deps)}++;
  open my $fh, "<", $file or fail("cannot read $file to check harness markers");
  my $s = do { local $/; <$fh> // "" };
  close $fh or fail("cannot close $file while checking harness markers");
  my $items = eval { parse_dune($s, $file) } or fail($@ =~ s/\n$//r);
  check_dune_markers($file, $dir, $items, $by_base, $seen, $in_deps);
}
if (%hand_run) {
  my %by_base = map { (s{.*/}{}r) => $_ } sort keys %hand_run;
  my %seen;
  for my $file (grep { m{(?:^|/)dune\z} && -f } @known) {
    read_dune_markers($file, dir_of($file), \%by_base, \%seen, 0);
  }
}

my (%whole, %some);
open my $rd, ">", "$tmp/harness-read" or fail("cannot write $tmp/harness-read");
for my $p (@changed) {
  my @hits;
  for my $h (@members) {
    # What the text reaches is every leg's; a declaration only adds the legs
    # it names for a path the text does not reach.
    my @d = grep { my $d = $_; grep { $p =~ $_ } @{$d->{re}} } @{$decl{$h} // []};
    if (exists $reach{$h}{$p}) {
      $whole{$h} = 1;
      push @hits, "$h (every leg: " . chain($h, $p) . ")";
    } elsif (@d) {
      my $at = "its affected-reads line" . (@d > 1 ? "s " : " ") . join(", ", map { $_->{line} } @d);
      if (grep { !defined $_->{ere} } @d) { $whole{$h} = 1; push @hits, "$h (every leg: $at)" }
      else {
        my %e = map { $_->{ere} => 1 } @d;
        $some{$h}{$_} = 1 for keys %e;
        push @hits, "$h (legs " . join(" ", map { "'$_'" } sort keys %e) . ": $at)";
      }
    }
  }
  if (@hits) {
    print $rd "$p\t" . scalar(@hits) . "\n";
    push @err, "affected: $p: read by " . scalar(@hits) . " harness(es): " . join("; ", @hits);
  } elsif ($hand_run{$p}) {
    push @err, "affected: $p: verified hand-run harness marker";
  } elsif (is_script($p)) {
    push @err, "affected: $p: no harness reads it (no member of tools/test-harnesses.sh --list names it, loads it,"
      . " runs a script loading it, or declares it)";
  }
}
close $rd;
open my $cm, ">", "$tmp/harness-cmds" or fail("cannot write $tmp/harness-cmds");
for my $h (sort grep { $whole{$_} || $some{$_} } @members) {
  if ($whole{$h}) { print $cm "$h\n"; next }
  my @e = sort keys %{$some{$h}};
  printf $cm "%s --only '%s'\n", $h, @e == 1 ? $e[0] : join("|", map { "($_)" } @e);
}
close $cm;
open my $er, ">", "$tmp/harness-err" or fail("cannot write $tmp/harness-err");
print $er "$_\n" for @err;
close $er;
open my $ee, ">", "$tmp/harness-eres" or fail("cannot write $tmp/harness-eres");
print $ee "$_\n" for @eres;
close $ee;
# The census (the header's): every member's reach, or MEMBER's, one line per
# reached path or declared glob.
exit 0 unless ($ARGV[1] // "") eq "census";
my $only = $ARGV[2];
fail("--harness-census $only: not a member of tools/test-harnesses.sh --list") if defined $only && !grep { $_ eq $only } @members;
my @census;
for my $h (grep { !defined $only || $_ eq $only } @members) {
  for my $p (keys %{$reach{$h}}) {
    my $s = $reach{$h}{$p};
    push @census, join "\t", $h, $p, defined $s ? ($s->[1], "$s->[0]:$s->[2]", $s->[3], $s->[4]) : ("self", "-", "-", "-");
  }
  for my $d (@{$decl{$h} // []}) {
    push @census, join "\t", $h, $_, "declares", "$h:$d->{line}", defined $d->{ere} ? "--only '$d->{ere}'" : "-", "-"
      for @{$d->{globs}};
  }
}
open my $cs, ">", "$tmp/census" or fail("cannot write $tmp/census");
print $cs "$_\n" for sort @census;
close $cs;
HARNESS
  census_args=()
  [ "$census" = 0 ] || census_args=(census ${census_member:+"$census_member"})
  perl "$tmp/harness.pl" "$tmp" ${census_args[@]+"${census_args[@]}"} || exit 2
  # The selectors as the harnesses will match them: bash's own `=~` (exit 2 an
  # expression it refuses), as scripts/harness-support.sh's harness_args checks.
  while IFS=$'\t' read -r ere where; do
    ere_rc=0
    [[ '' =~ $ere ]] || ere_rc=$?
    [ "$ere_rc" != 2 ] || die "$where: --only '$ere' is not an extended regular expression bash accepts"
  done <"$tmp/harness-eres"
else
  [ "$harnesses" = 0 ] || die "--harnesses: there is no tools/test-harnesses.sh here to list the harnesses"
  [ "$census" = 0 ] || die "--harness-census: there is no tools/test-harnesses.sh here to list the harnesses"
  echo "affected: no tools/test-harnesses.sh here, so no harness legs are mapped" >"$tmp/harness-err"
fi
if [ "$census" = 1 ]; then
  cat "$tmp/harness-err" >&2
  cat "$tmp/census"
  printf 'affected: harness census: %d line(s) over %d member(s)\n' "$(wc -l <"$tmp/census" | tr -d ' ')" \
    "$(cut -f1 "$tmp/census" | sort -u | wc -l | tr -d ' ')" >&2
  exit 0
fi
if [ "$harnesses" = 1 ]; then
  cat "$tmp/harness-err" >&2
  cat "$tmp/harness-cmds"
  printf 'affected: %d harness command(s) from %d changed path(s)\n' \
    "$(wc -l <"$tmp/harness-cmds" | tr -d ' ')" "$(sort -u "$tmp/changed" | sed '/^$/d' | wc -l | tr -d ' ')" >&2
  exit 0
fi
command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"

dune describe workspace --root . --with-pps >"$tmp/workspace" 2>"$tmp/err" ||
  die "dune describe workspace failed:
$(cat "$tmp/err")"
dune describe external-lib-deps --root . >"$tmp/lib-deps" 2>"$tmp/err" ||
  die "dune describe external-lib-deps failed:
$(cat "$tmp/err")"
# Every directory holding a dune file may define per-test aliases.
if [ "$in_git" = 1 ]; then
  git ls-files --cached --others --exclude-standard -- dune '*/dune' >"$tmp/dune-files" ||
    die "git ls-files failed"
else
  find . -name _build -prune -o -name dune -type f -print | sed 's|^\./||' >"$tmp/dune-files"
fi
while IFS= read -r f; do dirname -- "$f"; done <"$tmp/dune-files" | sort -u >"$tmp/dirs"
[ -s "$tmp/dirs" ] || die "no dune files found"
dirs=()
while IFS= read -r d; do dirs+=("$d"); done <"$tmp/dirs"

# The probe (gh-ocannl-1541). What an alias rule's own action reads, dune
# never prints: `dune rules @ALIAS` evaluates an alias's actions lazily, which
# records no dependency. So the rules are handed to dune in a form it does
# print, in a scratch copy of the sources: each dune file is replaced by
# dune's own canonical text (`dune format-dune-file`: dune's lexer, so `#|`,
# block strings and escapes are its), and each rule attached to an alias that
# declares no targets becomes a TWIN, the same rule writing a file of its own
# (`__affected_probe_<hash>`) with an `(alias (name X) (deps <twin>))` keeping
# its aliases defined. `dune rules` on a twin is what the action reads, as dune
# evaluates it: `%{dep:}`, what `diff` infers, `(:include)`, `chdir`, globs,
# needed aliases. A test without a golden gets an empty one, so dune makes its
# output rule (which shows the test's `(deps)` and `(action)`). The converter
# is an allowlist (convert.pl below); what it does not rewrite is named in the
# manifest and stays as blind as before. No probe (it failed, or
# OCANNL_TOOL_AFFECTED_PROBE=0) means every alias action is read as before.
# Dune here never sees the caller's build directory.
probe_dune() { # DIR ARGS...: dune run at the root of a probe tree
  local d=$1
  shift
  (cd "$d" && env -u DUNE_BUILD_DIR -u DUNE_ROOT dune "$@" --root .)
}
cat >"$tmp/convert.pl" <<'CONVERT'
# convert.pl ROOT MANIFEST FORMS RECURSE LIST: rewrite each dune file LIST names in the
# probe ROOT into dune's canonical text with alias rules turned into twins;
# append to MANIFEST what was rewritten and what was not, to FORMS each
# top-level form of FILE (for a dune-file edit's comparison). RECURSE=0 leaves
# `(include ...)`d fragments alone (they are converted already).
use strict; use warnings; use Digest::MD5 qw(md5_hex); use File::Spec;
my ($root, $manifest, $forms_out, $recurse, $list) = @ARGV;
open my $lh, "<", $list or die "cannot read $list\n";
my @files = grep { $_ ne "" } map { s/\r?\n\z//r } <$lh>;
open my $man, ">>", $manifest or die "cannot write $manifest\n";
open my $fo, ">>", $forms_out or die "cannot write $forms_out\n";
sub canonical {   # dune's own text of a dune file, or undef
  my ($path) = @_;
  my $pid = open(my $fh, "-|") // die "cannot fork\n";
  if (!$pid) { open STDERR, ">", File::Spec->devnull; exec "dune", "format-dune-file", $path; exit 127 }
  local $/; my $s = <$fh> // ""; close $fh;
  return $? == 0 ? $s : undef;
}
# dune's printed text read as lists of raw tokens: atoms and quoted strings kept
# as printed (re-emitted verbatim), comments dropped. Anything else (a raw
# newline in a string, say, which dune's printer escapes): undef.
sub forms_of {
  my ($s) = @_;
  my @stack = ([]);
  pos($s) = 0;
  while (pos($s) < length $s) {
    next if $s =~ /\G\s+/gc || $s =~ /\G;[^\n]*/gc;
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) { @stack > 1 or return undef; my $l = pop @stack; push @{$stack[-1]}, $l; next }
    if ($s =~ /\G("(?:[^"\\\n]++|\\.)*+")/gc) { push @{$stack[-1]}, $1; next }
    if ($s =~ /\G([^\s()";]+)/gc) { push @{$stack[-1]}, $1; next }
    return undef;
  }
  return @stack == 1 ? $stack[0] : undef;
}
sub text { my ($x) = @_; ref $x ? "(" . join(" ", map { text($_) } @$x) . ")" : $x }
sub named { my ($x, $n) = @_; ref $x && @$x && !ref $x->[0] && $x->[0] eq $n }
sub is_field { my ($x) = @_; ref $x && @$x && !ref $x->[0] }
sub name_lit { my ($x) = @_; !ref $x && $x =~ /^[A-Za-z0-9_.+-]+$/ }
sub path_in {   # DIR/P normalized, for a literal relative P staying in the tree, else undef
  my ($dir, $p) = @_;
  return undef if ref $p || $p !~ m{^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$};
  my @parts = $dir eq "." ? () : split m{/}, $dir;
  for my $c (split m{/}, $p) {
    next if $c eq ".";
    if ($c eq "..") { @parts or return undef; pop @parts } else { push @parts, $c }
  }
  return @parts ? join("/", @parts) : ".";
}
# The paths a stanza's fields name (see "Literal dependencies" in the header):
# an atom of its `(deps ...)`, one inside a `(file ...)` there
# or inside a `(:NAME ...)` group (as an atom or a `(file ...)`), and the PATH
# of every `%{dep:PATH}` token in any field. A literal relative path staying in
# the tree only (path_in): a glob, a `source_tree`, an alias, a variable or a
# quoted string names nothing.
sub dep_tokens {   # DIR X ACC: each `%{dep:PATH}` in X
  my ($dir, $x, $acc) = @_;
  if (ref $x) { dep_tokens($dir, $_, $acc) for @$x; return }
  for my $t ($x =~ /%\{dep:([^}%\s"]+)\}/g) { my $q = path_in($dir, $t); $acc->{$q} = 1 if defined $q }
}
sub literal_names {   # DIR FIELDS: the sorted paths those fields name
  my ($dir, $fl) = @_;
  my %n;
  my $lit = sub { for (grep { !ref } @_) { my $q = path_in($dir, $_); $n{$q} = 1 if defined $q && $q ne "." } };
  for my $f (grep { named($_, "deps") } @$fl) {
    for my $d (@$f[1 .. $#$f]) {
      if (!ref $d) { $lit->($d); next }
      if (named($d, "file")) { $lit->(@$d[1 .. $#$d]); next }
      next unless is_field($d) && $d->[0] =~ /^:/;
      for my $e (@$d[1 .. $#$d]) { if (!ref $e) { $lit->($e) } elsif (named($e, "file")) { $lit->(@$e[1 .. $#$e]) } }
    }
  }
  dep_tokens($dir, $fl, \%n);
  return sort keys %n;
}
my $unvouched = 0;   # notes saying a form is not described, so far
sub note { $unvouched++ if $_[0] =~ /^(?:blind|blindtree|undescribed)$/; print $man join("\t", @_), "\n" }
# Heads of stanzas that attach no action to an alias. One with an `(alias ...)`
# field (copy_files has one) makes the alias depend on what it makes: a
# `source`, whose rules `dune rules @ALIAS` prints. Tests are read from
# their names (their goldens are placed after), an inline-test library by its name.
my %plain = map { $_ => 1 } "executable", "executables", "library", "test", "tests", "copy_files", "copy_files#",
  "ocamllex", "ocamlyacc", "menhir", "install", "env", "dirs", "data_only_dirs", "vendored_dirs", "include_subdirs",
  "documentation", "generate_sites_module", "foreign_library";
my %written;
sub convert {   # DIR FILE FORMS TOP [OCC] -> the rewritten forms
  my ($dir, $file, $forms, $top, $occ) = @_;
  $occ //= {};
  my @out;
  # Each top-level form is recorded once it is processed (the continue block),
  # with whether the converter vouched for it: no note said otherwise.
  my ($record, $before);
  for my $f (@$forms) {
    ($record, $before) = (undef, $unvouched);
    unless (is_field($f)) { note("blind", $dir, "$file holds a top-level form with no stanza name"); push @out, $f; next }
    my ($h, @fl) = @$f;
    my $canon = text($f);
    if ($top) {
      my @names = grep { name_lit($_) } map { named($_, "name") || named($_, "names") ? @$_[1 .. $#$_] : () } @fl;
      my @attached = grep { name_lit($_) } map { named($_, "alias") || named($_, "aliases") ? @$_[1 .. $#$_] : () } @fl;
      $record = [$file, $dir, $h, join(",", @names), (grep { named($_, "action") } @fl) ? 1 : 0];
      push @$record, join(",", @attached);
      push @$record, $canon;
    }
    if ($h eq "subdir") {
      my $sub = @fl ? path_in($dir, $fl[0]) : undef;
      if (defined $sub) { push @out, [$h, $fl[0], convert($sub, $file, [@fl[1 .. $#fl]], 0, $occ)]; next }
      note("blindtree", $dir, "$file has a (subdir) whose directory is not a literal path below it"); push @out, $f; next;
    }
    if ($h eq "include") {
      my $frag = @fl == 1 && $top ? path_in($dir, $fl[0]) : undef;
      unless (defined $frag && -f "$root/$frag") { note("blind", $dir, "$file has an (include) this reader cannot follow"); push @out, $f; next }
      if ($recurse) {
        my $t = canonical("$root/$frag");
        my $ff = defined $t ? forms_of($t) : undef;
        unless ($ff) { note("blind", $dir, "dune's text of $frag is not readable here"); push @out, $f; next }
        my @c = convert($dir, $frag, $ff, 0);
        unless ($written{$frag}++) {
          open my $o, ">", "$root/$frag" or die "cannot write $root/$frag\n";
          print $o text($_), "\n\n" for @c;
          close $o;
        }
      }
      push @out, $f; next;
    }
    my @af = grep { named($_, "alias") || named($_, "aliases") } @fl;
    my @al = map { @$_[1 .. $#$_] } @af;
    if (grep { !name_lit($_) } @al) { note("blind", $dir, "$file has a $h whose alias name is not literal"); push @out, $f; next }
    # Dependencies found while the rule runs -- what an `(:include FILE)` adds,
    # read from FILE, or what a `dynamic-run` reports -- `dune rules` does not
    # print: a form with one is not vouched for, as an alias action or as the
    # rules of its directory.
    if ($canon =~ /(?:^|[\s(])(:include|dynamic-run)[\s)]/) {
      my $what = $1;
      if ($h eq "rule" && @al) { note("undescribed", $dir, $_, "a rule of $file with a $what, whose dependencies dune rules does not print") for @al }
      else { note("blind", $dir, "$file has a ($h) with a $what, whose dependencies dune rules does not print") }
      push @out, $f; next;
    }
    if ($h eq "rule") {
      unless (@al) { push @out, $f; next }
      if (grep { named($_, "targets") || named($_, "target") } @fl) { note("source", $dir, $_) for @al; push @out, $f; next }
      my @act = grep { named($_, "action") } @fl;
      if (@act != 1 || @{$act[0]} != 2 || grep { !is_field($_) } @fl) {
        note("undescribed", $dir, $_, "a rule of $file with no single action field") for @al;
        push @out, $f; next;
      }
      # Named by the file, the aliases and the rule's rank among theirs, never by
      # its text: an edit dune reads the same (quoting, layout) changes no rule.
      # Share $occ across subdir blocks so separate blocks targeting the same
      # directory and alias get distinct probe targets (gh-ocannl-2446).
      my $key = join(" ", @al);
      my $twin = "__affected_probe_" . substr(md5_hex(join("\0", $file, $key, ++$occ->{$key})), 0, 12);
      push @out, ["rule", map { named($_, "action") ? ["action", ["progn", ["write-file", $twin, '""'], $_->[1]]] : $_ }
                          grep { !named($_, "alias") && !named($_, "aliases") } @fl];
      push @out, ["alias", ["name", $_], ["deps", $twin]] for @al;
      note("twin", $dir, $twin, @al);
      my @lit = literal_names($dir, \@fl);
      note("names", $dir, $twin, @lit) if @lit;
      next;
    }
    if ($h eq "alias") {
      my @n = map { @$_[1 .. $#$_] } grep { named($_, "name") } @fl;
      if (@n != 1 || !name_lit($n[0])) { note("blind", $dir, "$file has an (alias) whose name is not literal"); push @out, $f; next }
      if (grep { named($_, "action") } @fl) {
        note("undescribed", $dir, $n[0], "an (alias) of $file with an action");
        push @out, $f; next;
      }
      note("source", $dir, $n[0]);
      my @deps = grep { named($_, "deps") } @fl;
      my @lit = grep { /\.sh\z/ } literal_names($dir, \@deps);
      unless (@lit && !grep { !is_field($_) || $_->[0] !~ /^(?:name|deps|enabled_if|package|locks)$/ } @fl) {
        push @out, $f; next;
      }
      # An alias expansion can combine several stanzas' reads. Give this
      # stanza its own rule so a sibling's dependency cannot vouch for it.
      # A file can contribute several blocks to one directory. An included
      # file uses the same target names in each directory that includes it.
      my $key = "alias $n[0]";
      my $twin = "__affected_probe_" . substr(md5_hex(join("\0", $file, $key, ++$occ->{$key})), 0, 12);
      push @out, ["rule", @deps, grep({ named($_, "enabled_if") || named($_, "package") } @fl),
                  ["action", ["write-file", $twin, '""']]];
      push @out, [$h, map { named($_, "deps") ? [@$_, $twin] : $_ } @fl];
      note("alias-witness", $dir, $twin, $n[0], @lit);
      next;
    }
    unless ($plain{$h}) { note("blind", $dir, "$file has a ($h) stanza, which this reader does not model"); push @out, $f; next }
    # A test's alias action is dune's documented one: its output rule (a golden
    # is placed below where it has none) diffed against its golden. An inline-test
    # library's runner action is not described.
    if ($h eq "test" || $h eq "tests" || ($h eq "library" && grep { named($_, "inline_tests") } @fl)) {
      my @nf = grep { named($_, "name") || named($_, "names") } @fl;
      my @n = map { @$_[1 .. $#$_] } @nf;
      if (!@n || grep { !name_lit($_) } @n) { note("blind", $dir, "$file has a ($h) whose names are not literal"); push @out, $f; next }
      if ($h eq "library") { note("undescribed", $dir, "runtest-$n[0]", "dune runs it as an inline-test action") }
      else { note("test", $dir, $_) for @n }
    }
    note("source", $dir, $_) for @al;
    push @out, $f;
  } continue {
    if ($record) { my ($attached, $canon) = splice @$record, -2; print $fo join("\t", @$record, $unvouched == $before ? 1 : 0, $attached, $canon), "\n" }
  }
  return @out;
}
for my $file (@files) {
  next unless -f "$root/$file";
  my ($dir) = $file =~ m{^(.*)/[^/]+$} ? ($1) : (".");
  my $t = canonical("$root/$file");
  my $forms = defined $t ? forms_of($t) : undef;
  unless ($forms) { note("blind", $dir, "dune's text of $file is not readable here"); next }
  my @c = convert($dir, $file, $forms, 1);
  open my $o, ">", "$root/$file" or die "cannot write $root/$file\n";
  print $o text($_), "\n\n" for @c;
  close $o;
}
CONVERT
# Tests, as the converter read them (their literal names): one with no golden,
# where no rule makes one, gets an empty golden, so dune makes its output rule.
cat >"$tmp/tests.pl" <<'TESTS'
use strict; use warnings;
my ($root, $targets, $manifest) = @ARGV;
open my $th, "<", $targets or die "cannot read $targets\n";
my (%made, $cur);
while (<$th>) { chomp; if (/^(.*):$/) { $cur = $1; next } $made{$cur eq "." ? $_ : "$cur/$_"} = 1 if defined $cur }
open my $mh, "<", $manifest or die "cannot read $manifest\n";
while (<$mh>) {
  chomp;
  my ($k, $dir, $name) = split /\t/;
  next unless $k eq "test";
  my $golden = ($dir eq "." ? "" : "$dir/") . "$name.expected";
  next if -e "$root/$golden" || $made{$golden} || !-d "$root/$dir";
  open my $o, ">", "$root/$golden" or die "cannot write $root/$golden\n"; close $o;
}
TESTS
make_tree() { # PROBE: a scratch copy of the sources, converted (dune not run there yet)
  local p=$1
  mkdir -p "$p" || return 1
  # Ignored files are copied too (a local config, a fixture a rule reads), but
  # not ignored directories (`_build`, logs): a rule needing one fails in the
  # probe, and the reading falls back.
  if [ "$in_git" = 1 ]; then
    { git ls-files -z --cached --others --exclude-standard &&
      git ls-files -z --others --ignored --exclude-standard --directory |
        perl -0ne 'print unless m{/\0?\z}'; } >"$tmp/probe-files" || return 1
  else
    find . -name _build -prune -o \( -type f -o -type l \) -print0 >"$tmp/probe-files" || return 1
  fi
  # A path git still indexes but the worktree deleted is not copied.
  perl -0ne 'chomp; print "$_\0" if -e $_ || -l $_' "$tmp/probe-files" >"$tmp/probe-list" || return 1
  tar --null -T "$tmp/probe-list" -cf - | (cd "$p" && tar -xf -) || return 1
  perl "$tmp/convert.pl" "$p" "$tmp/manifest" "$tmp/forms" 1 "$tmp/dune-files" || return 1
}
finish_probe() { # PROBE SUFFIX: its tests' goldens, and its listings
  local p=$1 s=$2
  probe_dune "$p" show targets "${dirs[@]}" >"$tmp/probe-targets$s" 2>&1 || return 1
  probe_dune "$p" show aliases "${dirs[@]}" >"$tmp/probe-aliases$s" 2>&1 || return 1
  perl "$tmp/tests.pl" "$p" "$tmp/probe-targets$s" "$tmp/manifest$s" 2>>"$tmp/err" || return 1
}
# A changed dune file is compared across revisions (the old probe below) when
# both sides have it and the worktree's copy is the right-hand side's; any other
# keeps the reach of its whole directory, and says why.
scoped=()
: >"$tmp/unscoped"
while IFS= read -r f; do
  case $f in dune | */dune) ;; *) continue ;; esac
  why=
  if [ "$mode" = files ]; then why="--files names no revision to compare it with"
  elif [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" = 0 ]; then why="no probe"
  elif [ ! -f "$f" ]; then why="it is deleted"
  elif ! git cat-file -e "$old_rev:$f" 2>/dev/null; then why="$old_rev has no such file"
  elif [ -n "$new_rev" ] && ! git show "$new_rev:$f" 2>/dev/null | cmp -s - "$f"; then
    why="the worktree's copy is not $new_rev's"
  fi
  if [ -n "$why" ]; then printf '%s\t%s\n' "$f" "$why" >>"$tmp/unscoped"; else scoped+=("$f"); fi
done <"$tmp/changed"
qroot=.
probe_ok=0
old_ok=0
for x in manifest forms manifest-old forms-old old-targets old-aliases scoped; do : >"$tmp/$x"; done
if [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" = 0 ]; then
  echo "affected: OCANNL_TOOL_AFFECTED_PROBE=0: no probe, so every alias action dune does not dump stays blind" >&2
elif make_tree "$tmp/probe"; then
  # The old probe: this one with each compared dune file at the old revision.
  if [ ${#scoped[@]} -gt 0 ]; then
    if cp -R "$tmp/probe" "$tmp/probe-old" && printf '%s\n' "${scoped[@]}" >"$tmp/scoped" &&
       while IFS= read -r f; do git show "$old_rev:$f" >"$tmp/probe-old/$f" || exit 1; done <"$tmp/scoped" &&
       perl "$tmp/convert.pl" "$tmp/probe-old" "$tmp/manifest-old" "$tmp/forms-old" 0 "$tmp/scoped" &&
       finish_probe "$tmp/probe-old" -old; then
      old_dirs=()
      while IFS= read -r f; do old_dirs+=("$(dirname -- "$f")"); done <"$tmp/scoped"
      if probe_dune "$tmp/probe-old" show targets "${old_dirs[@]}" >"$tmp/old-targets" 2>&1 &&
         probe_dune "$tmp/probe-old" show aliases "${old_dirs[@]}" >"$tmp/old-aliases" 2>&1; then
        old_ok=1
      fi
    fi
    if [ "$old_ok" = 0 ]; then
      while IFS= read -r f; do printf '%s\t%s\n' "$f" "the old probe failed: $(tail -1 "$tmp/err" 2>/dev/null)" >>"$tmp/unscoped"; done <"$tmp/scoped"
      : >"$tmp/scoped"
    fi
  fi
  # The probe answers for the worktree only if it lists the same aliases: dune
  # exits 0 after an error in these listings, so a probe it does not load (an
  # (include) of a file the copy lacks, say) is caught here and read without.
  if finish_probe "$tmp/probe" ""; then
    if probe_dune . show aliases "${dirs[@]}" >"$tmp/worktree-aliases" 2>&1 &&
       cmp -s "$tmp/worktree-aliases" "$tmp/probe-aliases"; then
      probe_ok=1
      qroot=$tmp/probe
    else
      echo "the probe does not define the aliases the worktree does: $(diff "$tmp/worktree-aliases" "$tmp/probe-aliases" | head -3 | tr '\n' ' ')" >>"$tmp/err"
    fi
  fi
fi
if [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" != 0 ] && [ "$probe_ok" = 0 ]; then
  echo "affected: the probe failed, so every alias action dune does not dump stays blind: $(tail -5 "$tmp/err" 2>/dev/null)" >&2
  if [ -s "$tmp/scoped" ]; then
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "the probe failed" >>"$tmp/unscoped"; done <"$tmp/scoped"
  fi
  for x in manifest forms manifest-old forms-old scoped; do : >"$tmp/$x"; done
fi

listings() { # Dune 3.24 prints these listings on stderr, and exits 0 even after an
  # error, so both streams are read together, and every line must have one of
  # the forms the reader below accepts.
  for what in aliases targets; do
    probe_dune "$qroot" show "$what" "${dirs[@]}" >"$tmp/$what" 2>&1 ||
      die "dune show $what failed:
$(cat "$tmp/$what")"
  done
}
listings

cat >"$tmp/main.pl" <<'PERL'
use strict; use warnings;
my ($tmp, $referenced_only, $probe_ok, $qroot, $old_root, $gpu_gated) = @ARGV;
require "$tmp/dune-source.pl";
# While a dune answer is being read, a refusal keeps a copy of it (see keep_raw).
our ($raw_text, $raw_what);
sub fail {
  my $m = $_[0];
  $m .= " (the whole answer is kept in " . keep_raw($raw_text, $raw_what) . ")" if defined $raw_text;
  print STDERR "affected: $m\n"; exit 2;
}
sub slurp { local $/; open my $fh, "<", $_[0] or fail("cannot read $_[0]"); my $s = <$fh>; $s }

# A reader for the sexps dune prints: lists, atoms, quoted strings. $at, when
# given, collects the byte offset of each top-level item; a failure names the
# offset and the text around it.
sub around { my ($s, $p) = @_; my $from = $p > 60 ? $p - 60 : 0; (substr($s, $from, 120) =~ s/\n/\\n/gr) }
sub parse_sexp {
  my ($s, $what, $at) = @_;
  my @stack = ([]);
  pos($s) = 0;
  while (pos($s) < length $s) {
    if ($s =~ /\G\s+/gc) { next }
    push @$at, pos($s) if $at && @stack == 1;
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) {
      @stack > 1 or fail("unbalanced output from $what at byte " . (pos($s) - 1) . ": ..." . around($s, pos($s) - 1) . "...");
      my $l = pop @stack; push @{$stack[-1]}, $l; next;
    }
    if ($s =~ /\G"((?:[^"\\]++|\\.)*+)"/gcs) {
      my $a = $1; $a =~ s/\\(.)/$1 eq "n" ? "\n" : $1/ge;
      push @{$stack[-1]}, $a; next;
    }
    if ($s =~ /\G([^\s()"]+)/gc) { push @{$stack[-1]}, $1; next }
    fail("unreadable output from $what at byte " . pos($s) . ": ..." . around($s, pos($s)) . "...");
  }
  @stack == 1 or fail("unbalanced output from $what (" . (@stack - 1) . " list(s) left open)");
  return $stack[0];
}
sub field { # a value from a list of (key value) pairs
  my ($fields, $key) = @_;
  for my $f (@$fields) { return $f->[1] if ref $f && @$f == 2 && !ref $f->[0] && $f->[0] eq $key }
  return undef;
}
sub items { my $v = $_[0]; defined $v && ref $v ? @$v : () }
sub atom { my ($v, $what) = @_; defined $v && !ref $v or fail("expected an atom in $what"); $v }
sub pair { my ($e, $what) = @_; ref $e && @$e == 2 && !ref $e->[0] or fail("unexpected entry in $what"); @$e }
sub dir_of { $_[0] =~ m{^(.*)/([^/]+)$} ? ($1, $2) : (".", $_[0]) }
sub nsort { sort { $a <=> $b } @_ }   # numeric; outside any lexical $a or $b

# Libraries and executables: ((root R) (build_context C) (KIND FIELDS)...).
my $ws = parse_sexp(slurp("$tmp/workspace"), "dune describe workspace");
@$ws == 1 && ref $ws->[0] or fail("dune describe workspace printed no workspace");
my $ctx;
my @st;          # { kind, dir, names, units => {NAME => 1}, to => [stanza index...] }
my %by_modkey;   # "DIR\0Module" -> [stanza indices]
my %lib_by_uid;  # local library uid -> stanza index
my %lib_by_name; # public or local library name -> stanza index
my %generated;   # generated module file -> [stanza indices owning it]
my @requires;
for my $e (@{$ws->[0]}) {
  my ($kind, $v) = pair($e, "dune describe workspace");
  next if $kind eq "root";
  if ($kind eq "build_context") { $ctx = atom($v, "build_context"); next }
  $kind =~ /^(executables|library)$/
    or fail("dune describe workspace entry \"$kind\" is outside this helper's boundary (executables, library)");
  defined $ctx or fail("dune describe workspace gave no build_context before its stanzas");
  next if $kind eq "library" && (field($v, "local") // "") ne "true";
  my ($inc) = items(field($v, "include_dirs"));
  # Dune names a stanza's object directory after it: DIR/.NAME.objs for a
  # library (NAME its local name), DIR/.NAME.eobjs for executables.
  my ($dir, $objname) = atom($inc, "include_dirs") =~ m{^\Q$ctx\E/(?:(.+)/)?\.([^/]+)\.e?objs/}
    or fail("include_dir \"$inc\" is not an object directory under $ctx");
  $dir //= ".";
  my @names = $kind eq "library" ? (atom(field($v, "name"), "name")) : map { atom($_, "names") } items(field($v, "names"));
  @names or fail("an executables entry without names in dune describe workspace");
  push @st, { kind => $kind, dir => $dir, names => \@names, units => { map { $_ => 1 } @names } };
  my $i = $#st;
  push @requires, [items(field($v, "requires"))];
  if ($kind eq "library") {
    $lib_by_uid{atom(field($v, "uid"), "uid")} = $i;
    $st[$i]{units}{$objname} = 1;
    for my $n ($names[0], $objname) {
      !defined $lib_by_name{$n} || $lib_by_name{$n} == $i or fail("two libraries answer to the name $n");
      $lib_by_name{$n} = $i;
    }
  }
  for my $m (items(field($v, "modules"))) {
    my $name = atom(field($m, "name"), "module name");
    next if $name =~ /__$/;   # alias modules dune generates
    $st[$i]{units}{lcfirst $name} = 1;
    for my $key ("impl", "intf") {
      for my $p (items(field($m, $key))) {
        my $rel = atom($p, $key) =~ s{^\Q$ctx\E/}{}r;
        my ($d) = dir_of($rel);
        # A module file with no source of that name is generated (ocamllex,
        # menhir, a (select) copy, a rule's output): its rule says from what.
        push @{$generated{$rel}}, $i unless -e $rel;
        # Its sources, for the link-only ranking: a preprocessed `X.pp.ml` stands for
        # `X.ml`; any other file with no source makes its code not all known.
        my $src = -e $rel ? $rel : $rel =~ s/\.pp(\.mli?)$/$1/r;
        # (A wrapped library's alias module, `NAME.ml-gen`, is dune's and names only its modules.)
        if (-e $src) { push @{$st[$i]{mods}{$name}}, $src }
        elsif ($rel !~ /\.ml-gen$/) { $st[$i]{gen_mod} = 1; push @{$st[$i]{gen_files}}, [$name, $rel] }
        my $k = "$d\0$name";
        push @{$by_modkey{$k}}, $i unless grep { $_ == $i } @{$by_modkey{$k} // []};
      }
    }
  }
}
for my $i (0 .. $#st) {   # a uid outside %lib_by_uid is an installed library
  $st[$i]{to} = [map { $lib_by_uid{$_} } grep { defined $lib_by_uid{$_} } @{$requires[$i]}];
}

# Every stanza with its project-library edges, ppx rewriters included:
# ((CONTEXT (ENTRY...))), each ENTRY (KIND FIELDS). `dune describe workspace`
# leaves preprocessing out of `requires` and omits `tests` stanzas, and this
# listing leaves out stanzas with no dependency at all, so the two are merged:
# an entry here adds its edges to the described stanza of the same kind,
# directory and names, and one with no such stanza is added -- a `tests`
# stanza, or one this build does not compile (an optional backend library),
# whose modules are then unknown.
my $ext = parse_sexp(slurp("$tmp/lib-deps"), "dune describe external-lib-deps");
@$ext == 1 or fail("dune describe external-lib-deps printed " . scalar(@$ext) . " contexts, expected one");
my (undef, $entries) = pair($ext->[0], "dune describe external-lib-deps");
my %described = map { join("\0", $st[$_]{kind}, $st[$_]{dir}, sort @{$st[$_]{names}}) => $_ } 0 .. $#st;
my @edges;   # [stanza index, internal_deps]
for my $e (@$entries) {
  my ($kind, $v) = pair($e, "dune describe external-lib-deps");
  $kind =~ /^(executables|tests|library)$/
    or fail("dune describe external-lib-deps entry \"$kind\" is outside this helper's boundary (executables, tests, library)");
  my @names = map { atom($_, "names") } items(field($v, "names"));
  my $dir = atom(field($v, "source_dir"), "source_dir");
  @names or fail("a $kind entry without names in $dir");
  my $i = $described{join("\0", $kind, $dir, sort @names)};
  unless (defined $i) {
    push @st, { kind => $kind, dir => $dir, names => \@names, units => { map { $_ => 1 } @names }, to => [],
                undescribed => $kind ne "tests" };
    $i = $#st;
    if ($kind eq "library") {
      !defined $lib_by_name{$names[0]} or fail("two libraries answer to the name $names[0]");
      $lib_by_name{$names[0]} = $i;
    }
  }
  push @edges, [$i, field($v, "internal_deps")];
}
my (%skipped_optional, %skipped_off_build);
for my $edge (@edges) {
  my ($i, $deps) = @$edge;
  for my $d (items($deps)) {
    my ($n, $k) = pair($d, "internal_deps");
    if (defined $lib_by_name{$n}) { push @{$st[$i]{to}}, $lib_by_name{$n} }
    elsif ($k eq "optional") { $skipped_optional{$n} = 1 }
    # A stanza this build does not compile (an optional backend library whose
    # platform package is absent) may name another such library by its local
    # name, which this listing does not give -- only the public one: on Linux,
    # `metal_backend` requires `metal_math_api_runtime` (gh-ocannl-1389). No
    # stanza this build compiles links either one, so the edge carries no reach.
    elsif ($st[$i]{undescribed}) { $skipped_off_build{"$n (required by " . label($i) . ")"} = 1 }
    else { fail(label($i) . " uses project library $n, which dune does not describe") }
  }
}

# Per-directory listings: "DIR:" headers, one name per line.
my @dir_list = grep { $_ ne "" } split /\n/, slurp("$tmp/dirs");
my %is_dir = map { $_ => 1 } @dir_list;
our $listing_soft = 0;   # an unexpected line returns undef instead of stopping the run
sub listing {   # FILE [DIR...]: the listing of the directories asked (all, by default)
  my ($what, @asked) = @_;
  @asked = @dir_list unless @asked;
  my (%by, $cur);
  $cur = $asked[0] if @asked == 1;
  for my $l (split /\n/, slurp("$tmp/$what")) {
    next if $l eq "";
    # A directory dune does not build (a data-only one, say) defines nothing.
    if ($l =~ /^Error: Directory (.*) does not exist\.$/ && $is_dir{$1}) { $cur = undef; next }
    if ($l =~ /^(.*):$/ && $is_dir{$1}) { $cur = $1; next }
    # A directory target is listed with a trailing slash.
    unless ($l =~ s{^([^\s/]+)/?$}{$1}) { return undef if $listing_soft; fail("unexpected line from dune show $what: $l") }
    defined $cur or fail("dune show $what printed $l outside any directory");
    push @{$by{$cur}}, $l;
  }
  return \%by;
}
my $aliases = listing("aliases");
my $targets = listing("targets");
sub in_dir { my ($d, $f) = @_; $d eq "." ? $f : "$d/$f" }
# A `tests` stanza naming no library is in neither describe answer, but its
# executable is a target of its directory: one named by no stanza there is
# taken as such a stanza, linking nothing of the project.
for my $d (sort keys %$targets) {
  for my $x (map { /^(.+)\.exe$/ ? $1 : () } @{$targets->{$d}}) {
    next if grep { $st[$_]{dir} eq $d && $st[$_]{kind} ne "library" && grep { $_ eq $x } @{$st[$_]{names}} } 0 .. $#st;
    push @st, { kind => "tests", dir => $d, names => [$x], units => { $x => 1 }, to => [] };
  }
}

# What the probe's converter rewrote and what it did not (see convert.pl): per
# directory, the twins of each alias, the aliases with a stanza that makes them
# depend on files (`source`), and what stays blind.
my (%twins, %sources, %undescribed, %blind, %blindtree, %tested, %names_of, %alias_witness);
for (split /\n/, slurp("$tmp/manifest")) {
  my ($k, $d, @r) = split /\t/;
  if ($k eq "twin") { my $t = shift @r; push @{$twins{$d}{$_}}, in_dir($d, $t) for @r }
  elsif ($k eq "source") { $sources{$d}{$r[0]} = 1 }
  elsif ($k eq "undescribed") { $undescribed{$d}{$r[0]} //= $r[1] }
  elsif ($k eq "blind") { $blind{$d} //= $r[0] }
  elsif ($k eq "blindtree") { $blindtree{$d} //= $r[0] }
  elsif ($k eq "test") { $tested{$d}{"runtest-$r[0]"} = 1 }
  elsif ($k eq "names") { my $t = shift @r; $names_of{in_dir($d, $t)}{$_} = 1 for @r }
  elsif ($k eq "alias-witness") {
    my ($t, $a, @paths) = @r;
    $alias_witness{in_dir($d, $t)} = ["\@" . in_dir($d, $a), { map { $_ => 1 } @paths }];
  }
  else { fail("unexpected manifest line from the probe: $_") }
}
# Only the declaring alias's expansion can use a witness as evidence.
for my $d (keys %$targets) {
  @{$targets->{$d}} = grep { !$alias_witness{in_dir($d, $_)} } @{$targets->{$d}};
}
sub blind_reason {   # why an alias of DIR named NAME is not described, or undef
  my ($d, $name) = @_;
  return $blind{$d} if defined $blind{$d};
  for my $r (sort keys %blindtree) { return $blindtree{$r} if under($d, $r) }
  return $undescribed{$d}{$name} if defined $undescribed{$d}{$name};
  return undef if $twins{$d}{$name} || $sources{$d}{$name} || $tested{$d}{$name};
  return "no stanza this reader read attaches it";
}

# Per-test aliases, their name owners and their own targets.
my @al;   # { name, dir, owners => {index => 1}, own => [target...] }
my $artifact = qr/\.(?:exe|bc|bc-for-jsoo|js|cmi|cmo|cmx|cmt|cmti|cma|cmxa|cmxs|a|o|so|dll|pp\.ml|pp\.mli|ml-gen|d)$/;
for my $d (sort keys %$aliases) {
  my @in = grep { $st[$_]{dir} eq $d } 0 .. $#st;
  my %listed = map { $_ => 1 } @{$targets->{$d} // []};
  for my $a (@{$aliases->{$d}}) {
    my ($rest) = $a =~ /^(?:runtest|train|slow)-(.+)$/ or next;
    my ($unit, @variant) = split /-/, $rest;
    my @prefix = @variant ? ("$rest.", join(".", $unit, @variant) . ".") : ("$rest.");
    my @mine = grep { my $t = $_; $t !~ $artifact && grep { index($t, $_) == 0 } @prefix } @{$targets->{$d} // []};
    # `UNIT_*` targets count too (a rule-made `X_actual.ml` beside `runtest-X-ppx`,
    # test/ppx), though only a golden named as above makes the action a known diff.
    my @near = grep { $_ !~ $artifact && index($_, "${unit}_") == 0 } @{$targets->{$d} // []};
    my %own = map { in_dir($d, $_) => 1 } @mine, @near;
    # A golden pair diffs its golden against an output a rule makes. A golden
    # with no rule-made target beside it is diffed inside the alias's own action
    # (the run-and-diff shape: `runtest-kernel_attribution`, a `slow-` rule),
    # whose `(deps)` dune never dumps -- so that alias is opaque, not golden
    # (gh-ocannl-1431: an edit to the benchmarks/gh1002_cells.py such a rule
    # names in its `(deps)` missed it).
    my $golden = grep({ /\.expected$/ } @mine) && grep { !/\.expected$/ && !-e in_dir($d, $_) } @mine, @near;
    # With the probe, an alias's actions are its twins (a disabled rule makes
    # none) and its tests' golden diffs: described, unless the converter said
    # otherwise. Without it, a golden pair is the one action known.
    my $why;
    if ($probe_ok) {
      $why = blind_reason($d, $a);
      $own{$_} = 1 for grep { $listed{(dir_of($_))[1]} } @{$twins{$d}{$a} // []};
    }
    push @al, { name => "\@" . in_dir($d, $a), dir => $d, alias => $a,
                described => $probe_ok ? !defined $why : $golden ? 1 : 0, blind_why => $why,
                owners => { map { $_ => 1 } grep { $st[$_]{units}{$unit} } @in }, own => [sort keys %own] };
  }
}

# The rules of those targets: what they run and what they read.
# dune's own build directory is never a source, even where git does not ignore it.
my @changed = grep { $_ ne "" && !m{^_build(?:/|$)} } map { s{^\./}{}r } split /\n/, slurp("$tmp/changed");
# C-family stubs: the objects dune compiles (`X.o`) and the libraries' stub
# archives (`libNAME_stubs.a`) that bundle them, in every directory: an
# object's inputs can be generated elsewhere (a header a generator writes).
my @stub_targets = map { my $d = $_; map { in_dir($d, $_) } grep { /\.o$/ || /^lib.+_stubs\.a$/ } @{$targets->{$d} // []} }
  sort keys %$targets;
my @own_targets = sort { $a cmp $b } keys %{{ map { $_ => 1 } (map { @{$_->{own}} } @al), keys %generated, @stub_targets }};
my %exe;   # "DIR\0NAME" -> stanza indices that build that executable
for my $i (grep { $st[$_]{kind} ne "library" } 0 .. $#st) {
  push @{$exe{"$st[$i]{dir}\0$_"}}, $i for @{$st[$i]{names}};
}
sub glob_regex {   # dune's glob syntax: * ? [set] [!set] {a,b} \x; anything else is literal
  my ($g) = @_;
  my ($re, $depth) = ("", 0);
  while (length $g) {
    if ($g =~ s/^\[(!?)([^\]]+)\]//) {   # a class; [!set] is its complement
      my ($neg, $set) = ($1, $2);
      $re .= "[" . ($neg ? "^" : "") . ($set =~ s/([\\\[\]^])/\\$1/gr) . "]";
      next;
    }
    my $c = substr($g, 0, 1, "");
    if ($c eq "\\" && length $g) { $re .= quotemeta substr($g, 0, 1, "") }   # an escaped character
    elsif ($c eq "*") { $re .= "[^/]*" } elsif ($c eq "?") { $re .= "[^/]" }
    elsif ($c eq "{") { $re .= "(?:"; $depth++ } elsif ($c eq "}" && $depth) { $re .= ")"; $depth-- }
    elsif ($c eq "," && $depth) { $re .= "|" }
    else { $re .= quotemeta $c }
  }
  return $depth ? qr/./ : qr/^$re$/;   # unbalanced: matches anything
}
# A glob, or (and (P...)), (or (P...)), (not P), as a pair of matchers over
# file names: [may, must], what some reading of the printed predicate matches
# and what every reading does. The list wants "may"; `not` swaps the two.
# `dune rules` prints a glob with its escapes dropped (`\{x\}*.md` as
# `{x}*.md`), so which of its metacharacters were literal is lost. The
# readings of a printed glob: as a glob; with its braces, brackets and commas
# literal and its wildcards live; and as its own text, every character
# literal. An escaped wildcard is covered by the live one, an escaped brace or
# bracket by the second reading.
sub predicate_pair {
  my ($p) = @_;
  if (!ref $p) {
    my $re = glob_regex($p);
    my $plain = glob_regex($p =~ s/([{}\[\],])/\\$1/gr);
    return [sub { $_[0] eq $p || $_[0] =~ $re || $_[0] =~ $plain },
            sub { $_[0] eq $p && $_[0] =~ $re && $_[0] =~ $plain }];
  }
  my ($op, $arg) = pair($p, "a glob predicate");
  if ($op eq "not") { my ($may, $must) = @{predicate_pair($arg)}; return [sub { !$must->($_[0]) }, sub { !$may->($_[0]) }] }
  $op =~ /^(?:and|or)$/ && ref $arg or fail("a glob predicate \"$op\" is outside this helper's boundary");
  my @m = map { predicate_pair($_) } @$arg;
  my $all = sub { my ($k, $f) = @_; !grep { !$_->[$k]->($f) } @m };
  my $any = sub { my ($k, $f) = @_; grep { $_->[$k]->($f) } @m };
  return $op eq "and" ? [sub { $all->(0, $_[0]) }, sub { $all->(1, $_[0]) }]
                      : [sub { $any->(0, $_[0]) }, sub { $any->(1, $_[0]) }];
}
sub predicate { predicate_pair($_[0])->[0] }
sub strip_ctx { my ($p) = @_; $p =~ s{^\Q$ctx\E/}{} ? $p : undef }
# `dune rules` over many targets, in batches whose command line stays far below
# the smallest limit (Windows' 32767 characters for the whole command line).
# OCANNL_TOOL_AFFECTED_ARGS overrides the 16000-character budget, for tests.
my $arg_budget = $ENV{OCANNL_TOOL_AFFECTED_ARGS} // 16000;
$arg_budget =~ /^[1-9][0-9]*$/ or fail("OCANNL_TOOL_AFFECTED_ARGS must be a positive integer");
# The tree queried is $rules_root: the probe when there is one.
our $rules_root = $qroot;
our $rules_soft = 0;   # a failure returns undef instead of stopping the run
sub dune_rules {
  my @todo = @_;
  my @rules;
  while (@todo) {
    my ($len, @batch) = (0);
    while (@todo && (!@batch || $len + length($todo[0]) + 3 < $arg_budget)) { $len += length($todo[0]) + 3; push @batch, shift @todo }
    # Only the file `-o` names is the answer. To print a rule, dune builds what
    # the rule's dependencies are computed from (a `%{read:...}` file, say), and
    # what those actions print reaches dune's own streams: on a cold build
    # directory that was `Wrote value of 'backend_extension' to
    # ocannl_backend_extension.txt` in the middle of the sexps (gh-ocannl-1389).
    # Both streams are kept for the failure message, never parsed.
    unlink "$tmp/rules-out";
    my $pid = fork // fail("cannot fork for dune rules");
    if (!$pid) {
      open STDOUT, ">", "$tmp/rules-log" or exit 126;
      open STDERR, ">&", \*STDOUT or exit 126;
      chdir $rules_root or exit 126;
      delete @ENV{qw(DUNE_BUILD_DIR DUNE_ROOT)};
      exec "dune", "rules", "--root", ".", "-o", "$tmp/rules-out", @batch; exit 127;
    }
    unless (waitpid($pid, 0) == $pid && $? == 0) {
      return undef if $rules_soft;
      fail("dune rules failed:\n" . slurp("$tmp/rules-log"));
    }
    my $text = slurp("$tmp/rules-out");
    # A refusal from here on keeps what dune wrote, so its cause can be read
    # rather than guessed.
    local ($raw_text, $raw_what) = ($text, "dune-rules");
    my @at;
    my @got = @{parse_sexp($text, "dune rules", \@at)};
    for my $k (0 .. $#got) {
      ref $got[$k] or fail("unexpected atom in dune rules output: \"$got[$k]\" at byte $at[$k]: ..." . around($text, $at[$k]) . "...");
    }
    push @rules, @got;
  }
  return unstage(@rules);
}
# Dune 3.25 hands an action what `(package PKG)` needs as files of a staging
# area, `_build/install/CTX/.packages/HASH/PATH`, whose rules `dune rules`
# cannot print (gh-ocannl-2216; 3.24 printed the `.PKG-files` alias instead,
# read through `dune rules @ALIAS` below). Each is a link to the shared install
# area's `_build/install/CTX/PATH`, itself a link to the build file it
# installs: a staged file is read as that link's dependencies.
sub unstage {
  my @rules = @_;
  my (%link, @ask);   # staged file -> the shared area's link
  for my $rule (@rules) {
    for my $dep (items(field($rule, "deps"))) {
      next unless ref $dep && @$dep == 2 && !ref $dep->[0] && $dep->[0] eq "File";
      my ($where, $path) = pair($dep->[1], "a File dependency");
      next unless $where eq "In_build_dir" && $path =~ m{^(_build/install/[^/]+)/\.packages/[^/]+/(.+)$};
      push @ask, "$1/$2" unless $link{$path};
      $link{$path} = "$1/$2";
    }
  }
  return @rules unless @ask;
  my @linked = dune_rules(@ask);
  return undef if grep { !defined } @linked;
  my %via;   # the shared area's link -> what its rule needs
  for my $rule (@linked) {
    my $tg = field($rule, "targets");
    $via{atom($_, "targets")} = [items(field($rule, "deps"))] for items(field($tg // [], "files"));
  }
  for my $rule (@rules) {
    my $deps = field($rule, "deps") // next;
    @$deps = map {
      my $l = ref $_ && $_->[0] eq "File" && ref $_->[1] && $_->[1][0] eq "In_build_dir" ? $link{$_->[1][1]} : undef;
      !defined $l ? $_ : $via{$l} ? @{$via{$l}} : fail("dune rules printed no rule for $l, which $_->[1][1] stages")
    } @$deps;
  }
  return @rules;
}
# A copy of an answer this reader refused, outside the temporary directory the
# exit trap removes.
sub keep_raw {
  my ($text, $what) = @_;
  my $dir = $ENV{TMPDIR} // "/tmp";
  my $f = "$dir/affected-$what.$$.txt";
  open my $o, ">", $f or return "(could not write $f)";
  print $o $text; close $o;
  return $f;
}
sub has_opaque { my ($x) = @_; ref $x ? scalar(grep { has_opaque($_) } @$x) : $x =~ /^\x01/ }
sub canon {   # a stanza or form as text, for messages
  my ($x) = @_;
  return "(" . join(" ", map { canon($_) } @$x) . ")" if ref $x;
  return "\"" . ($x =~ s/(["\\])/\\$1/gr =~ s/\n/\\n/gr) . "\"";
}
sub dfield {   # the arguments of a stanza's first (NAME ...) field
  my ($x, $name) = @_;
  for my $f (@$x[1 .. $#$x]) { return [@$f[1 .. $#$f]] if ref $f && @$f && !ref $f->[0] && $f->[0] eq $name }
  return undef;
}
sub dfields {   # all (NAME ...) fields of a stanza -> ([args], ...)
  my ($x, $name) = @_;
  return map { [@$_[1 .. $#$_]] } grep { ref $_ && @$_ && !ref $_->[0] && $_->[0] eq $name } @$x[1 .. $#$x];
}
# Stanzas of DIR, `(subdir ...)` flattened into the directory it names.
sub dune_stanzas {
  my ($dir, $items) = @_;
  my @out;
  for my $x (@$items) {
    next unless ref $x && @$x && !ref $x->[0];
    if ($x->[0] eq "subdir" && @$x >= 2 && !ref $x->[1]) {
      my $sub = rpath($dir, $x->[1]);
      if (defined $sub) { push @out, dune_stanzas($sub, [@$x[2 .. $#$x]]); next }
    }
    push @out, { dir => $dir, kind => $x->[0], x => $x };
  }
  return @out;
}
sub rpath {   # BASE/P normalized, repository-relative; undef when it leaves the repository
  my ($base, $p) = @_;
  return undef if $p =~ m{^/};
  my @parts = $base eq "." ? () : split m{/}, $base;
  for my $c (split m{/}, $p) {
    next if $c eq "" || $c eq ".";
    if ($c eq "..") { @parts or return undef; pop @parts } else { push @parts, $c }
  }
  return @parts ? join("/", @parts) : ".";
}
my @gpu_stanzas; # stanza-local backend markers, kept out of the build reading
my @rule_backend_stanzas; # rules with explicit ocannl-backend: markers
my %default_defs; # default name -> { backends => [...], targets => [[Mod, def], ...], has_gpu => bool, where => str }
my @default_readers; # stanzas with ocannl-backend-default: reader markers
my @dstanzas;   # every stanza of every dune file, as read from the worktree
my %dune_text;  # dune file -> its text
my @backend_marker_files;
for my $f (grep { $_ ne "" } split /\n/, slurp("$tmp/dune-files")) {
  next unless -f $f;
  $dune_text{$f} = slurp($f);
  my $items = eval { parse_dune($dune_text{$f}, $f) } or fail($@ =~ s/\n$//r);
  push @dstanzas, dune_stanzas((dir_of($f))[0], $items);
  my $has_default = index($dune_text{$f}, "ocannl-backend-default:") >= 0;
  my $has_backend = index($dune_text{$f}, "ocannl-backend:") >= 0;
  if ($has_default || ($gpu_gated && $has_backend)) {
    my $mitems = eval { parse_dune($dune_text{$f}, $f, 1) } or fail($@ =~ s/\n$//r);
    for my $ds (dune_stanzas((dir_of($f))[0], $mitems)) {
      push @gpu_stanzas, $ds if $gpu_gated && defined dfield($ds->{x}, "affected_gpu_backend");
      push @rule_backend_stanzas, $ds if $ds->{kind} eq "rule" && defined dfield($ds->{x}, "affected_rule_backends");
      for my $def (dfields($ds->{x}, "affected_default_def")) {
        my ($dname, $bstr, $tstr, $where) = @$def;
        fail("$where: duplicate `ocannl-backend-default:` definition for `$dname` (already defined at $default_defs{$dname}{where})")
          if exists $default_defs{$dname};
        my @bks = split /,/, $bstr;
        my @tgs = map { [ split /\./, $_, 2 ] } split /,/, $tstr;
        $default_defs{$dname} = {
          backends => \@bks,
          targets  => \@tgs,
          has_gpu  => (grep { /^(?:cuda|hip|metal)$/ } @bks) ? 1 : 0,
          where    => $where,
          dir      => $ds->{dir},
          ds       => $ds,
        };
      }
      my @reads = dfields($ds->{x}, "affected_default_read");
      if (@reads) {
        if ($ds->{kind} eq "library" && !defined dfield($ds->{x}, "inline_tests")) {
          fail("$reads[0][1]: `ocannl-backend-default:` reader belongs on a test, executable, or inline-test library stanza, not a plain `(library)`");
        }
        my (%seen_r, @names_r);
        for my $r (@reads) {
          my ($dname, $where) = @$r;
          fail("$where: duplicate `ocannl-backend-default:` reader `$dname` on the same stanza")
            if $seen_r{$dname}++;
          push @names_r, [$dname, $where];
        }
        my @sbks = map { @$_ } dfields($ds->{x}, "affected_stanza_backends");
        push @default_readers, {
          dir => $ds->{dir}, kind => $ds->{kind}, x => $ds->{x}, reads => \@names_r,
          stanza_bks => @sbks ? { map { $_ => 1 } @sbks } : undef,
        };
      }
    }
  } elsif ($has_backend) {
    push @backend_marker_files, $f;
  }
}
# If any backend-default markers exist, also read explicit rule backend markers from other files so
# variant rules pinned to a disjoint backend do not inherit an owner's default marker.
if (%default_defs || @default_readers) {
  for my $f (@backend_marker_files) {
    my $mitems = eval { parse_dune($dune_text{$f}, $f, 1) } or fail($@ =~ s/\n$//r);
    for my $ds (dune_stanzas((dir_of($f))[0], $mitems)) {
      push @rule_backend_stanzas, $ds if $ds->{kind} eq "rule" && defined dfield($ds->{x}, "affected_rule_backends");
    }
  }
  for my $dr (@default_readers) {
    for my $r (@{$dr->{reads}}) {
      my ($dname, $where) = @$r;
      fail("$where: `ocannl-backend-default:` reader names `$dname`, which no `(library)` stanza defines")
        unless exists $default_defs{$dname};
    }
  }
}
# An edit to a fragment a dune file `(include ...)`s (or `dynamic_include`s)
# reaches like an edit to that dune file.
my %fragment_dirs;   # fragment -> every directory whose dune file includes it
for my $s (grep { $_->{kind} =~ /^(?:dynamic_)?include$/ } @dstanzas) {
  my $frag = $s->{x}[1];
  my $p = !ref $frag && $frag !~ /%\{/ ? rpath($s->{dir}, $frag) : undef;
  push @{$fragment_dirs{$p}}, $s->{dir} if defined $p;
}

# Each queried target's rule: the executables it runs, the files and globs it
# reads, and the generated files it reads (`via`), whose rules are queried in
# turn -- a `copy_files` copy, a file list -- so what a rule reaches through
# them counts as its own. Compiled objects and executables are not followed:
# an executable is reached through its stanza.
# A compiled artifact a rule reads directly -- a stanza's `.NAME.objs` or
# `.NAME.eobjs` directory, or an installed library's files -- makes that
# stanza an owner of the reader, as an executable it runs does.
sub compiled_owners {
  my ($path) = @_;   # a build path, context prefix included
  if ($path =~ m{^_build/install/[^/]+/lib/([^/]+)(?:/([^/]+))?/}) {
    my @n = grep { defined $lib_by_name{$_} } (defined $2 ? ("$1.$2") : ()), $1;
    return @n ? ($lib_by_name{$n[0]}) : ();
  }
  my $rel = strip_ctx($path) // return ();
  my ($d, $x, $e) = $rel =~ m{^(?:(.*)/)?\.([^/]+)\.(e?)objs(?:/|$)} or return ();
  $d //= ".";
  return grep { $st[$_]{dir} eq $d && ($e ? $st[$_]{kind} ne "library" : $st[$_]{kind} eq "library")
                && $st[$_]{units}{$x} } 0 .. $#st;
}
my (%runs, %reads, %globs, %via, %adeps, %copies_one);
my %queried;
# An alias some stanza makes depend on files (a rule with targets, an
# `(alias)` stanza, `copy_files`): `dune rules @ALIAS` expands those
# dependencies (not actions), helper aliases included, so what it prints is
# the alias's own too. One query per alias, since one answer for several does
# not say whose each rule is.
sub source_query {   # ALIAS [FORCE]: FORCE asks dune whatever the manifest says
  my ($a, $force) = @_;
  return unless $probe_ok && $a->{described} && ($force || $sources{$a->{dir}}{$a->{alias}});
  local $rules_soft = 1;
  my @got = dune_rules($a->{name});
  if (grep { !defined } @got) {
    $a->{described} = 0;
    $a->{blind_why} = "dune rules $a->{name} failed";
    return;
  }
  my %own = map { $_ => 1 } @{$a->{own}};
  for my $rule (@got) {
    my $tg = field($rule, "targets");
    my @t = map { atom($_, "targets") } map { items(field($tg // [], $_)) } "files", "directories";
    if (my @rel = grep { defined } map { strip_ctx($_) } @t) {
      for my $rel (@rel) {
        my $witness = $alias_witness{$rel};
        unless ($witness) { $own{$rel} = 1; next }
        # A glob or a helper alias can pick up a witness. Its rule is evidence
        # only for the alias whose stanza declared those dependencies.
        my ($alias, $paths) = @$witness;
        next unless $alias eq $a->{name};
        for my $dep (items(field($rule, "deps"))) {
          next unless ref $dep && $dep->[0] eq "File";
          my ($where, $path) = pair($dep->[1], "a File dependency");
          my $p = $where eq "In_build_dir" ? strip_ctx($path) : $where eq "In_source_tree" ? $path : undef;
          $a->{named_deps}{$p} = 1 if defined $p && $paths->{$p};
        }
      }
      next;
    }
    # An installed file (what `(package PKG)` needs) is a link to a file of the
    # build: that file is the alias's own, or a library's compiled artifact.
    for my $dep (items(field($rule, "deps"))) {
      my ($k, $v) = ref $dep ? @$dep : ($dep);
      next if $k =~ /^(?:Env|Universe|Sandbox_config)$/;
      my ($where, $path) = $k eq "File" ? pair($v, "a File dependency") : ("", "");
      if ($where eq "In_build_dir" && (my @o = compiled_owners($path))) { $a->{owners}{$_} = 1 for @o }
      elsif ($where eq "In_build_dir" && defined(my $rel = strip_ctx($path))) { $own{$rel} = 1 unless $alias_witness{$rel} }
      elsif ($where ne "External") {
        $a->{described} = 0;
        $a->{blind_why} = "$a->{name} needs @t, whose rule this reader does not follow";
      }
    }
  }
  $a->{own} = [sort keys %own];
}
source_query($_) for @al;
@own_targets = sort { $a cmp $b } keys %{{ map { $_ => 1 } @own_targets, map { @{$_->{own}} } @al }};
sub query_rules {   # TARGET...: 1, or 0 when dune could not print them (under $rules_soft)
my @queue = @_;
while (@queue) {
  my @got = dune_rules(@queue);
  return 0 if grep { !defined } @got;
  $queried{$_} = 1 for @queue;
  my %next;
  for my $rule (@got) {
    my $tg = field($rule, "targets");
    my @files = map { atom($_, "targets") } map { items(field($tg // [], $_)) } "files", "directories";
    my (%r, %p, @g, %v, %ad);
    for my $dep (items(field($rule, "deps"))) {
      next if !ref $dep && $dep eq "Universe";   # always reruns; reads nothing in particular
      ref $dep && @$dep && !ref $dep->[0] or fail("unexpected dependency in dune rules output");
      my $k = $dep->[0];
      if ($k eq "File") {
        my ($where, $path) = pair($dep->[1], "a File dependency");
        if ($where eq "In_build_dir") {
          if (my @o = compiled_owners($path)) { $r{$_} = 1 for @o; next }
          my $rel = strip_ctx($path) // next;   # another context's, or _build/install
          next if $alias_witness{$rel};
          if ($rel =~ m{^(?:(.*)/)?([^/]+)\.(?:exe|bc)$}) { $r{$_} = 1 for @{$exe{($1 // ".") . "\0$2"} // []} }
          else {
            $p{$rel} = 1;
            $v{$rel} = 1 if !-e $rel && $rel !~ $artifact && $rel !~ m{(?:^|/)\.[^/]+\.e?objs/};
          }
        } elsif ($where eq "In_source_tree") { $p{$path} = 1 }
        elsif ($where ne "External") { fail("a File dependency \"$where\" is outside this helper's boundary") }
      } elsif ($k eq "glob") {
        my $spec = $dep->[1];
        my ($where, $path) = pair(field($spec, "dir"), "a glob dependency");
        next if $where eq "External";
        if ($where eq "In_build_dir" and my @o = compiled_owners("$path/")) { $r{$_} = 1 for @o; next }
        next if (field($spec, "only_generated_files") // "false") eq "true";
        my $gd = $where eq "In_build_dir" ? strip_ctx($path) : $path;
        next unless defined $gd;
        push @g, [$gd, predicate(field($spec, "predicate"))];
      } elsif ($k eq "Alias") {
        # Another alias this rule needs: when that one is listed, so is this.
        my $spec = $dep->[1];
        my ($where, $path) = pair(field($spec, "dir"), "an Alias dependency");
        my $ad = $where eq "In_build_dir" ? strip_ctx($path) : $path;
        defined $ad or ($path eq $ctx and $ad = ".") or next;
        $ad{"\@" . in_dir($ad, atom(field($spec, "name"), "alias name"))} = 1;
      } elsif ($k !~ /^(?:Env|Universe|Sandbox_config)$/) {
        fail("a dependency \"$k\" in dune rules output is outside this helper's boundary");
      }
    }
    # A copy (`copy`, `copy#`, a `(select)` arm's copy-line-directive), its
    # `chdir` read past: what copy_source vouches for.
    my $act = field($rule, "action");
    $act = $act->[2] while ref $act && @$act == 3 && !ref $act->[0] && $act->[0] eq "chdir";
    my $is_copy = ref $act && @$act && !ref $act->[0]
      && ($act->[0] =~ /^copy#?$/ || ($act->[0] eq "ext" && @$act == 2 && !ref $act->[1] && $act->[1] =~ /^"?\((?:copy-line-directive|copy#?)\s/));
    for my $f (@files) {
      my $t = strip_ctx($f) // next;
      $runs{$t} = \%r; $reads{$t} = \%p; $globs{$t} = \@g; $via{$t} = [sort keys %v]; $adeps{$t} = \%ad;
      $copies_one{$t} = 1 if $is_copy;
    }
    $next{$_} = 1 for grep { !$queried{$_} } keys %v;
  }
  for (@queue) { exists $runs{$_} or $rules_soft ? return 0 : fail("dune rules printed no rule for $_") }
  @queue = sort keys %next;
}
return 1;
}
# Under the probe, a rule dune cannot print there (an input its copy lacks, say)
# sends the whole reading back to the shell, which reads again without it.
sub probe_query {
  local $rules_soft = $probe_ok;
  return if query_rules(@_);
  print STDERR "affected: dune could not print a rule in the probe (an input its copy lacks?); reading again without it\n";
  exit 4;
}
probe_query(@own_targets);
# A helper alias a queried rule needs (`(deps (alias NAME))`, NAME no per-test
# alias) is an entry of its own, never printed: its twins are queried and read
# like a per-test alias's, and an alias needing it is listed when it would be
# (see the tiers below), to a fixpoint.
{
  my %is_al = map { $_->{name} => 1 } @al;
  for (my $grew = 1; $grew;) {
    $grew = 0;
    my %need = map { %{$adeps{$_} // {}} } sort keys %queried;
    for my $n (sort grep { !$is_al{$_} } keys %need) {
      $is_al{$n} = 1;
      my ($d, $x) = $n =~ m{^\@(?:(.*)/)?([^/]+)$} or next;
      $d //= ".";
      my %listed = map { $_ => 1 } @{$targets->{$d} // []};
      # One dune makes itself (`.PKG-files`, what `(package PKG)` needs) is
      # dependencies only, which `dune rules @ALIAS` expands; one dune does not
      # list otherwise is not known.
      my $internal = $x =~ /^\./;
      my $why = !$probe_ok ? "no probe" : $internal ? undef
              : grep({ $_ eq $x } @{$aliases->{$d} // []}) ? blind_reason($d, $x) : "dune does not list it";
      my $h = { name => $n, dir => $d, alias => $x, hidden => 1, described => !defined $why, blind_why => $why,
                owners => {}, own => [grep { $listed{(dir_of($_))[1]} } @{$twins{$d}{$x} // []}] };
      push @al, $h;
      source_query($h, $internal);
      probe_query(grep { !$queried{$_} } @{$h->{own}});
      $grew = 1;
    }
  }
}
# A rule-made file with the name of a file it reads, in another directory, is
# a copy of it (`copy_files`): an opaque action there may read the copy.
my %copies;   # path -> [directories holding a copy of it]
for my $t (sort keys %queried) {
  my ($td, $tb) = dir_of($t);
  for my $src (keys %{$reads{$t}}) {
    my ($sd, $sb) = dir_of($src);
    push @{$copies{$src}}, $td if $sb eq $tb && $sd ne $td;
  }
}
{ # Fold each target's generated inputs into it (cycle-safe: a file is folded once).
  my %done;
  sub fold {
    my ($t) = @_;
    return if $done{$t}++;
    for my $u (@{$via{$t}}) {
      fold($u);
      $runs{$t} = { %{$runs{$t}}, %{$runs{$u}} };
      $reads{$t} = { %{$reads{$t}}, %{$reads{$u}} };
      $globs{$t} = [@{$globs{$t}}, @{$globs{$u}}];
      $adeps{$t} = { %{$adeps{$t}}, %{$adeps{$u}} };
    }
  }
  fold($_) for sort keys %queried;
}
# A generated module file that is a plain copy -- its rule's action is a copy
# (a `(select)` arm, `copy#`), it runs nothing and reads one OCaml source -- has
# that source as its code (gh-ocannl-1765: a select arm made a test's code "not
# all known"). Any other action (a `sed`, an `echo`) may write what its input
# never names.
sub copy_source {   # generated module file -> its one source, or undef
  my ($g) = @_;
  return undef if !$copies_one{$g} || !$reads{$g} || %{$runs{$g} // {}} || @{$globs{$g} // []} || %{$adeps{$g} // {}}
    || @{$via{$g} // []};
  my @src = keys %{$reads{$g}};
  return @src == 1 && $src[0] =~ /\.mli?\z/ && -f $src[0] ? $src[0] : undef;
}
for my $s (grep { $_->{gen_files} } @st) {
  my @src = map { copy_source($_->[1]) } @{$s->{gen_files}};
  next if grep { !defined } @src;
  push @{$s->{mods}{$s->{gen_files}[$_][0]}}, $src[$_] for 0 .. $#src;
  $s->{gen_mod} = 0;
}

# A stub object belongs to the library whose stub archive (`libNAME_stubs.a`,
# NAME its local name) reads it. One no archive reads is linked into an
# executable, and which one dune does not say without computing link rules.
my %obj_libs;   # object path -> {library stanza index}
for my $t (grep { /(?:^|\/)lib.+_stubs\.a$/ } @stub_targets) {
  my ($d, $b) = dir_of($t);
  my ($name) = $b =~ /^lib(.+)_stubs\.a$/;
  my @libs = grep { $st[$_]{kind} eq "library" && $st[$_]{dir} eq $d && $st[$_]{units}{$name} } 0 .. $#st;
  for my $o (grep { /\.o$/ } keys %{$reads{$t} // {}}) { $obj_libs{$o}{$_} = 1 for @libs }
}

# A generated module's stanza is reached through what its rule reads (a
# `.mll`, a `.mly`, a (select) choice) and through the executable it runs.
for my $g (keys %generated) {
  for my $o (@{$generated{$g}}) { push @{$st[$o]{to}}, keys %{$runs{$g}}; push @{$st[$o]{gen_to}}, keys %{$runs{$g}} }
}
# A stub object's owners: the library whose stub archive bundles it, or else
# every executable and `tests` stanza of its directory. They are reached
# through the executables its rule runs, generated inputs folded in (a
# generator writing a header the object includes).
sub obj_owners {
  my ($o) = @_;
  my @libs = sort keys %{$obj_libs{$o} // {}};
  my ($od) = dir_of($o);
  return @libs ? @libs : grep { $st[$_]{kind} ne "library" && $st[$_]{dir} eq $od } 0 .. $#st;
}
for my $o (grep { /\.o$/ } @stub_targets) {
  for my $w (obj_owners($o)) { push @{$st[$w]{to}}, keys %{$runs{$o}}; push @{$st[$w]{gen_to}}, keys %{$runs{$o}} }
}

my (%hit, %dune_dirs, %data_dirs, %read_by, @unmapped, @unread, %unmapped_mod);
# The changed paths a harness reads (gh-ocannl-1900), and how many harnesses.
my %harness_read;
if (open my $hr, "<", "$tmp/harness-read") {
  while (<$hr>) { chomp; my ($p, $n) = split /\t/; $harness_read{$p} = $n }
  close $hr;
}
# For `--why` (gh-ocannl-1711), the first changed path behind each of those:
# a hit stanza's, a dune directory's, a data directory's, any data at all, and
# per changed path and alias the own target whose rule reads it.
my (%hit_why, %hit_path, %dune_why, %data_why, %read_via, %direct_why, %direct_path);
my $any_data_why;
sub hit_reason {
  my ($i, $path, $why) = @_;
  $hit_path{$i} //= $path;
  $hit_why{$i} //= $why;
}
# How a stanza was hit: by changed modules it owns (NAME => 1), or as a whole
# (a generated module, a stub, a changed stanza, a tests directory's modules).
my (%mod_hit, %whole_hit, %mod_paths);
my $any_data = 0;
sub label { my $s = $st[$_[0]]; "$s->{kind} @{$s->{names}} ($s->{dir})" }
# A glob entry is [DIR, matcher over a file name, recursive?]; a recursive one
# (glob_files_rec, source_tree) matches in DIR and beneath it.
sub reads {
  my ($t, $p, $d, $b) = @_;
  $reads{$t}{$p} || grep { ($_->[2] ? under($d, $_->[0]) : $_->[0] eq $d) && $_->[1]->($b) } @{$globs{$t}};
}
sub under { my ($x, $root) = @_; $root eq "." || $x eq $root || index($x, "$root/") == 0 }

# A changed dune file compared across revisions (gh-ocannl-1541): the old
# probe is this one with the file at the old revision. What it changed is
# what dune says changed -- the rules of its directory's targets, twins
# included, that differ between the probes -- plus, from dune's canonical text
# of both sides, the stanzas whose compilation the rules shown do not cover
# (flags live in the object rules): a changed library, executables or tests
# stanza, named by its literal names as dune lists them here. Any other
# changed form (`env`, `subdir`, `include`, `dirs`, an unknown head, an
# `(alias)` with an action), or a changed rule this reader cannot map to an
# owner, keeps the reach of the whole directory beneath the file, and says why.
my (%covered, %direct, %delta_paths);
my %unscoped = map { my ($f, $w) = split /\t/; ($f => $w) } grep { $_ ne "" } split /\n/, slurp("$tmp/unscoped");
my @scoped = grep { $_ ne "" } split /\n/, slurp("$tmp/scoped");
if (@scoped) {
  my @sdirs = map { (dir_of($_))[0] } @scoped;
  my ($old_aliases, $old_targets);
  { local $listing_soft = 1; $old_aliases = listing("old-aliases", @sdirs); $old_targets = listing("old-targets", @sdirs) }
  unless ($old_aliases && $old_targets) {
    $unscoped{$_} = "dune does not load the old revision's dune file beside these sources (see dune show in the old probe)" for @scoped;
    @scoped = ();
  }
  my %twin_of;   # twin path -> [aliases], either side
  my %old_sources;   # DIR -> {alias some stanza of the old side made depend on files}
  my %old_exp;
  sub old_expansion {   # DIR ALIAS: {target => 1} that `dune rules @ALIAS` gives in the old probe
    my ($d, $x) = @_;
    return $old_exp{"$d\0$x"} //= do {
      local $rules_root = $old_root;
      local $rules_soft = 1;
      my @got = dune_rules("\@" . in_dir($d, $x));
      # Unanswered: every target counts as its own.
      if (grep { !defined } @got) { +{ "\0all" => 1 } }
      else {
        my %t;
        for my $rule (@got) {
          my $tg = field($rule, "targets");
          $t{$_} = 1 for grep { defined } map { strip_ctx(atom($_, "targets")) } map { items(field($tg // [], $_)) } "files", "directories";
        }
        \%t;
      }
    };
  }
  for my $m ("manifest", "manifest-old") {
    for (split /\n/, slurp("$tmp/$m")) {
      my ($k, $d, $t, @a) = split /\t/;
      push @{$twin_of{in_dir($d, $t)}}, @a if $k eq "twin";
      $old_sources{$d}{$t} = 1 if $k eq "source" && $m eq "manifest-old";
    }
  }
  my %forms;   # side -> file -> [form records]
  for my $side ("new", "old") {
    for (split /\n/, slurp($side eq "new" ? "$tmp/forms" : "$tmp/forms-old")) {
      my ($file, $d, $head, $names, $act, $vouched, $attached, $canon) = split /\t/, $_, 8;
      push @{$forms{$side}{$file}}, { dir => $d, head => $head, names => [grep { $_ ne "" } split /,/, $names],
                                      action => $act, vouched => $vouched, canon => $canon,
                                      attached => [grep { $_ ne "" } split /,/, $attached] };
    }
  }
  # Dune omits tests' modules from its workspace description. For compared
  # dune files only, literal module lists (or the default test names) on either
  # side supply their owners. Unsupported ordered sets stay unknown. This maps
  # new .pp.ml/.pp.mli rules without excusing genuinely ownerless rules, and
  # keeps a module added to an existing test attached to that whole stanza.
  for my $side ("new", "old") {
    for my $file (sort keys %{$forms{$side} // {}}) {
      for my $r (grep { $_->{vouched} && $_->{head} =~ /^(?:test|tests)$/ } @{$forms{$side}{$file}}) {
        my $x = eval { parse_dune($r->{canon}, "a form of $file") };
        next unless $x && @$x == 1 && ref $x->[0];
        my $mods = dfield($x->[0], "modules") // $r->{names};
        next if grep { ref $_ || !/^[A-Za-z][A-Za-z0-9_']*$/ } @$mods;
        my @ix = grep { $st[$_]{kind} eq "tests" && $st[$_]{dir} eq $r->{dir}
                        && join("\0", sort @{$st[$_]{names}}) eq join("\0", sort @{$r->{names}}) } 0 .. $#st;
        for my $m (@$mods) {
          my $k = "$r->{dir}\0" . ucfirst $m;
          for my $i (@ix) { push @{$by_modkey{$k}}, $i unless grep { $_ == $i } @{$by_modkey{$k} // []} }
        }
      }
    }
  }
  # The `(alias)` stanzas of this side, by directory and name: what an alias a
  # stanza makes depend on files may need is read off its stanzas' tokens, to a
  # fixpoint (a mention errs towards "needs it"; a template means anything).
  my (%alias_forms, %alias_dirs);
  for my $file (sort keys %{$forms{new} // {}}) {
    for my $r (grep { $_->{head} eq "alias" } @{$forms{new}{$file}}) {
      for my $n (@{$r->{names}}) { push @{$alias_forms{"$r->{dir}\0$n"}}, $r->{canon}; $alias_dirs{$n}{$r->{dir}} = 1 }
    }
  }
  sub mentions {   # DIR X H SEEN: may an (alias) stanza of X in DIR need the alias H?
    my ($sd, $x, $h, $seen) = @_;
    return 0 if $seen->{"$sd\0$x"}++;
    my @texts = @{$alias_forms{"$sd\0$x"} // []};
    # No (alias) stanza: its dependencies come from rules, whose needs dune prints.
    return 0 unless @texts;
    my @tok = grep { $_ ne "" } map { split /[\s()]+/ } @texts;
    return 1 if grep { /%\{/ || $_ eq $h || m{/\Q$h\E$} } @tok;
    for my $t (@tok) {
      my ($b) = $t =~ m{([^/]+)$} or next;
      for my $od (sort keys %{$alias_dirs{$b} // {}}) { return 1 if mentions($od, $b, $h, $seen) }
    }
    return 0;
  }
  my %rules_shown = map { $_ => 1 } "rule", "copy_files", "copy_files#", "ocamllex", "ocamlyacc", "menhir";
  my %kind_of = (test => "tests", tests => "tests", executable => "executables", executables => "executables", library => "library");
  my $per_test = qr/^(?:runtest|train|slow)-/;
  # A rule as text, each side's own probe root made one placeholder: the old
  # root extends the new one (`.../probe-old`), so one pattern over both would
  # leave `<probe>-old` on the old side (review round 1).
  my %root_re = (old => qr/\Q$old_root\E/, new => qr/\Q$qroot\E/);
  sub rule_text { my ($x) = @_; ref $x ? "(" . join(" ", map { rule_text($_) } @$x) . ")" : $x }
  FILE: for my $f (@scoped) {
    my ($d) = dir_of($f);
    my (@why, @parts, %whole, %direct_here, @paths);
    # The changed forms: each side's canonical forms not matched on the other.
    my %attached;
    my (%count, @changed_forms);
    $count{$_->{canon}}++ for @{$forms{new}{$f} // []};
    $count{$_->{canon}}-- for @{$forms{old}{$f} // []};
    for my $side ("new", "old") {
      my %left = map { $_ => ($side eq "new" ? $count{$_} : -$count{$_}) } keys %count;
      for my $r (@{$forms{$side}{$f} // []}) { push @changed_forms, { %$r, side => $side } if $left{$r->{canon}}-- > 0 }
    }
    # A library form that changed only in which modules it holds and which
    # libraries it links, by literal names (a split, a merge, a module moved
    # between libraries, with no text of a module changed; gh-ocannl-1651), is
    # reached by its modules rather than as a whole: what it links runs the same
    # code, so its users are ranked by naming one of its modules, either side's.
    # A library new or gone on one side counts when its form holds only such
    # fields (and `name`, `public_name`, `package`, `synopsis`). Any other field
    # that differs -- flags, preprocessing, stubs, `(select)`, `(re_export)` --
    # keeps the stanza whole.
    my %lib_forms;   # side -> library name -> its parsed form, undef when unread
    for my $r (grep { $_->{head} eq "library" } @changed_forms) {
      my $x = eval { parse_dune($r->{canon}, "a form of $f") };
      $lib_forms{$r->{side}}{$_} = $x && @$x == 1 && ref $x->[0] ? $x->[0] : undef for @{$r->{names}};
    }
    my $form_fields = sub {   # parsed form -> {FIELD => field}, or undef for a form not read as fields
      my ($x) = @_;
      my %h;
      for my $fl (@$x[1 .. $#$x]) {
        return undef unless ref $fl && @$fl && !ref $fl->[0] && !exists $h{$fl->[0]};
        $h{$fl->[0]} = $fl;
      }
      \%h;
    };
    my $literal = sub { my ($fl) = @_; !$fl || !grep { ref $_ || /^[:\\]|%\{/ } @$fl[1 .. $#$fl] };
    my $membership_only = sub {   # changed library form -> 1 when only its membership differs
      my ($r) = @_;
      my $other = $r->{side} eq "new" ? "old" : "new";
      my ($mine) = grep { defined } map { $lib_forms{$r->{side}}{$_} } @{$r->{names}};
      my $f1 = $mine && $form_fields->($mine) or return 0;
      my ($theirs) = grep { defined } map { $lib_forms{$other}{$_} } @{$r->{names}};
      return 0 if !$theirs && grep { exists $lib_forms{$other}{$_} } @{$r->{names}};   # there, but unread
      if ($theirs) {
        my $f2 = $form_fields->($theirs) or return 0;
        for my $k (sort keys %{{ %$f1, %$f2 }}) {
          next if canon($f1->{$k} // []) eq canon($f2->{$k} // []);
          return 0 unless ($k eq "modules" || $k eq "libraries") && $literal->($f1->{$k}) && $literal->($f2->{$k});
        }
        return 1;
      }
      return 0 if !$f1->{modules} || grep { !/^(?:name|public_name|package|modules|libraries|synopsis)$/ } keys %$f1;
      return $literal->($f1->{modules}) && $literal->($f1->{libraries}) ? 1 : 0;
    };
    my (%split, %split_mods);   # stanza index -> 1; -> {Module => 1} its old side held
    my $split_lib = sub { my ($sd, $n) = @_; scalar grep { $split{$_} && $st[$_]{dir} eq $sd && $st[$_]{units}{$n} } 0 .. $#st };
    for my $r (@changed_forms) {
      my $h = $r->{head};
      # A form the converter did not vouch for (an alias rule it left alone, an
      # (alias) with an action, a stanza outside its allowlist): what changed
      # there no rule this reader compares may show.
      unless ($r->{vouched}) { push @why, "a changed ($h) the probe does not describe"; next }
      # The aliases the changed forms attach to (their `alias`/`aliases` fields),
      # counted per side: an alias depends on a rule's targets, not the other
      # way, so moving a rule between aliases changes no rule dune prints.
      $attached{$r->{side}}{in_dir($r->{dir}, $_)}++ for @{$r->{attached}};
      next if $rules_shown{$h};
      if ($h eq "alias" && !$r->{action}) {
        # The alias itself, per-test or a helper (whose entry carries the change to
        # the aliases whose rules need it); and, since `dune rules @ALIAS` folds a
        # helper's dependencies into an alias some stanza makes depend on files
        # without saying which came through it, every such alias too.
        for my $x (@{$r->{names}}) {
          $direct_here{in_dir($d, $x)} = "its (alias) stanza changed";
          next if $x =~ $per_test;
          for my $sd (sort keys %sources) {
            $direct_here{in_dir($sd, $_)} = "it may depend on the changed (alias) $x"
              for grep { $_ =~ $per_test && mentions($sd, $_, $x, {}) } keys %{$sources{$sd}};
          }
        }
        next;
      }
      if (my $kind = $kind_of{$h}) {
        unless (@{$r->{names}}) { push @why, "a changed ($h) with no literal name"; next }
        for my $n (@{$r->{names}}) {
          my @ix = grep { $st[$_]{dir} eq $d && $st[$_]{kind} eq $kind
                          && ($kind eq "library" ? $st[$_]{units}{$n} : grep { $_ eq $n } @{$st[$_]{names}}) } 0 .. $#st;
          if (@ix && $kind eq "library" && $membership_only->($r)) {
            $split{$_} = 1 for @ix;
            my ($x) = grep { defined } map { $lib_forms{$r->{side}}{$_} } @{$r->{names}};
            for my $i (@ix) { $split_mods{$i}{ucfirst $_} = 1 for grep { !ref } @{dfield($x, "modules") // []} }
            next;
          }
          if (@ix) { $whole{$_} = 1 for @ix; next }
          # Gone from this side: nothing of it is left to run.
          my $gone = $r->{side} eq "old" && !grep({ $_ eq "$n.exe" } @{$targets->{$d} // []})
                     && !grep { $_ eq "runtest-$n" } @{$aliases->{$d} // []};
          push @why, "a changed ($h) named $n, which dune does not list here" unless $gone;
        }
        next;
      }
      push @why, $h eq "alias" ? "a changed (alias) with an action" : "a changed ($h) stanza";
    }
    for my $x (sort keys %{{ %{$attached{old} // {}}, %{$attached{new} // {}} }}) {
      $direct_here{$x} //= "the stanzas attached to it changed"
        if ($attached{old}{$x} // 0) != ($attached{new}{$x} // 0);
    }
    unless (@why) {
      # The rules of every target of the directory, on both sides.
      my %text;
      for my $side ("old", "new") {
        my @t = map { in_dir($d, $_) } @{($side eq "old" ? $old_targets : $targets)->{$d} // []};
        local $rules_root = $side eq "old" ? $old_root : $qroot;
        local $rules_soft = 1;
        my @got = @t ? dune_rules(@t) : ();
        if (grep { !defined } @got) { push @why, "dune rules failed on the $side side"; last }
        for my $rule (@got) {
          my $s = rule_text($rule);
          $s =~ s/$root_re{$side}/<probe>/g;
          my $tg = field($rule, "targets");
          $text{$side}{$_} = $s for grep { defined } map { strip_ctx(atom($_, "targets")) } map { items(field($tg // [], $_)) } "files", "directories";
        }
      }
      my @delta = grep { ($text{old}{$_} // "") ne ($text{new}{$_} // "") } sort keys %{{ %{$text{old} // {}}, %{$text{new} // {}} }};
      for my $t (@why ? () : @delta) {
        my ($td, $tb) = dir_of($t);
        my $new_side = exists $text{new}{$t};
        if ($tb =~ /^__affected_probe_/) {
          $direct_here{in_dir($td, $_)} = "a rule of it changed" for @{$twin_of{$t} // []};
        } elsif ($tb =~ /^(.+)\.(?:exe|bc)$/) {
          my @ix = @{$exe{"$td\0$1"} // []};
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, which no stanza here builds" if !@ix && $new_side;
        } elsif ($tb =~ /^([^.]+)\.pp\.mli?$/) {
          my @ix = @{$by_modkey{"$td\0" . ucfirst $1} // []};
          # A module moved into a library that preprocesses differently: the module
          # is what changed, and it ranks by its name (below).
          next if @ix && !grep { !$split{$_} } @ix;
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, whose module no stanza here owns" if !@ix && $new_side;
        } elsif ($tb =~ /\.o$/) {
          $whole{$_} = 1 for obj_owners($t);
        } elsif ($tb =~ /^(?:lib(.+)_stubs\.a|(.+)\.(?:cmxa|cma|cmxs|a))$/) {
          my ($n, $stubs) = ($1 // $2, defined $1);
          my @ix = grep { $st[$_]{kind} eq "library" && $st[$_]{dir} eq $td && $st[$_]{units}{$n} } 0 .. $#st;
          # A library's archive lists its modules: a membership change changes it.
          next if !$stubs && @ix && !grep { !$split{$_} } @ix;
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, which no library here makes" if !@ix && $new_side;
        } elsif ($tb =~ /^(.+?)(?:__)?\.ml(?:-gen)?$/ && $split_lib->($td, $1)) {
          # The alias module dune writes for a wrapped library (`LIB__.ml-gen`, or
          # `LIB.ml-gen` when no module is named after it) names its modules.
          next;
        } elsif ($tb =~ $artifact) {
          push @why, "a changed rule for $t, an artifact this reader does not map" if $new_side;
        } else {
          push @paths, $t;
          # A target only the old side makes is read nowhere in this side's graph:
          # the aliases that may have owned it there (named after it, or made to
          # depend on files by an old stanza) are listed.
          unless ($new_side) {
            for my $a (grep { $_->{dir} eq $td && !$_->{hidden} } @al) {
              my ($rest) = $a->{alias} =~ /^(?:runtest|train|slow)-(.+)$/ or next;
              my ($unit, @var) = split /-/, $rest;
              next unless grep({ index($tb, $_) == 0 } "$rest.", "${unit}_", join(".", $unit, @var) . ".")
                || ($old_sources{$td}{$a->{alias}} && grep { $_ } @{old_expansion($td, $a->{alias})}{$t, "\0all"});
              $direct_here{in_dir($td, $a->{alias})} = "a target it may have owned is gone";
            }
          }
        }
      }
      # A per-test alias new on this side is listed: it never ran.
      my %was = map { $_ => 1 } @{$old_aliases->{$d} // []};
      $direct_here{in_dir($d, $_)} = "it is new" for grep { $_ =~ $per_test && !$was{$_} } @{$aliases->{$d} // []};
      push @parts, scalar(@delta) . " changed rule(s)";
    }
    if (@why) {
      $unscoped{$f} = join("; ", @why);
      next FILE;
    }
    $covered{$f} = 1;
    # A stanza whose modules are not all known (a generated one) has no names to
    # rank it by: whole.
    $whole{$_} = 1 for grep { $st[$_]{gen_mod} || !$st[$_]{mods} } keys %split;
    my @by_membership = grep { !$whole{$_} } sort { $a <=> $b } keys %split;
    for my $i (@by_membership) {
      $hit{$i} = 1;
      hit_reason($i, $f, "$f (a compared dune edit changed only its modules or libraries)");
      for my $m (keys %{$st[$i]{mods}}, keys %{$split_mods{$i} // {}}) { $mod_hit{$i}{$m} = 1; $mod_paths{$m}{"\0membership"} = 1 }
    }
    $hit{$_} = $whole_hit{$_} = 1 for keys %whole;
    hit_reason($_, $f, "$f (a compared dune edit changed the stanza as a whole)") for keys %whole;
    $direct{"\@$_"} //= $direct_here{$_} for keys %direct_here;
    $direct_why{"\@$_"} //= "$f, compared with the old revision: $direct_here{$_}" for keys %direct_here;
    $direct_path{"\@$_"} //= $f for keys %direct_here;
    $delta_paths{$_} //= $f for @paths;
    push @parts, scalar(keys %whole) . " stanza(s) changed as a whole" if %whole;
    push @parts, scalar(@by_membership) . " library stanza(s) changed only in their modules or libraries (" .
      join(", ", map { label($_) } @by_membership) . "), reached by their modules" if @by_membership;
    push @parts, scalar(keys %direct_here) . " alias(es) whose own rules changed" if %direct_here;
    print STDERR "affected: $f: compared with the old revision: " . join(", ", @parts) . "\n";
  }
}
print STDERR "affected: $_: its whole directory, because $unscoped{$_}\n" for sort keys %unscoped;

# Literal dependencies (see the header). For dependency-only aliases,
# source_query checks the witness without adding it to the ordinary graph.
sub names_literally {   # ALIAS PATH
  my ($a, $p) = @_;
  return 0 unless $a->{described};
  return 1 if $a->{named_deps} && $a->{named_deps}{$p};
  return grep { $names_of{$_} && $names_of{$_}{$p} && $reads{$_} && $reads{$_}{$p} } @{$a->{own}};
}
sub runs_sweep_driver {   # ALIAS
  my ($a) = @_;
  return 0 unless $a->{described};
  for my $t (@{$a->{own}}) {
    for my $si (keys %{$runs{$t} // {}}) {
      my $s = $st[$si];
      return 1 if $s->{dir} eq "test/operations" && $s->{kind} eq "executables"
        && grep { $_ eq "sweep_harness_driver" } @{$s->{names}};
    }
  }
  return 0;
}

# Only a directory wholly gated by literal system comparisons can account for
# an otherwise ownerless module. Optional libraries with missing dependencies
# and directories with mixed or opaque guards keep the ordinary UNMAPPED path.
my (%stanzas_here, %opaque_ancestor);
for my $s (@dstanzas) {
  push @{$stanzas_here{$s->{dir}}}, $s;
  $opaque_ancestor{$s->{dir}} = 1 if $s->{kind} =~ /\A(?:include|dynamic_include|include_subdirs)\z/;
}
my ($system, $asked_system, %disabled_here);
sub disabled_stanzas {
  my ($d) = @_;
  return $disabled_here{$d} if exists $disabled_here{$d};
  my @forms = @{$stanzas_here{$d} // []};
  return $disabled_here{$d} = [] unless @forms;
  # An ancestor's include can introduce more stanzas through (subdir ...).
  return $disabled_here{$d} = [] if grep { under($d, $_) } keys %opaque_ancestor;
  my @gates;
  for my $s (@forms) {
    return $disabled_here{$d} = [] unless $s->{kind} =~ /\A(?:library|executables?|tests?|rule|alias)\z/;
    my $g = dfield($s->{x}, "enabled_if");
    return $disabled_here{$d} = [] unless $g && @$g == 1 && ref $g->[0] && @{$g->[0]} == 3
      && !grep { ref } @{$g->[0]};
    my ($op, $left, $right) = @{$g->[0]};
    ($left, $right) = ($right, $left) if $right eq '%{system}';
    return $disabled_here{$d} = [] unless $op eq '=' && $left eq '%{system}' && $right =~ /\A[a-zA-Z0-9_]+\z/;
    push @gates, $right;
  }
  unless ($asked_system++) {
    if (open my $fh, "-|", "dune", "exec", "--root", ".", "--no-build", "--", "printf", '%s\n', 'system=%{system}') {
      my @answer = <$fh>;
      $system = $1 if close($fh) && @answer == 1 && $answer[0] =~ m{(?:\A|/)system=([a-zA-Z0-9_]+)\n\z};
    }
  }
  return $disabled_here{$d} = [] unless defined $system && !grep { $_ eq $system } @gates;
  return $disabled_here{$d} = \@forms;
}

for my $p (@changed) {
  my ($d, $b) = dir_of($p);
  for my $a (@al) {
    for my $t (@{$a->{own}}) { if (reads($t, $p, $d, $b)) { $read_by{$p}{$a->{name}} = 1; $read_via{$p}{$a->{name}} //= $t; last } }
  }
  my $n_read = keys %{$read_by{$p} // {}};
  my (@ix, @parts, $unmapped, $by_module);
  my @gen = grep { reads($_, $p, $d, $b) } sort keys %generated;
  if (@gen) {
    my %seen_g;
    my @g = grep { !$seen_g{$_}++ } map { @{$generated{$_}} } @gen;
    $hit{$_} = $whole_hit{$_} = 1 for @g;
    hit_reason($_, $p, "$p (the rule generating " . join(", ", @gen) . " reads it)") for @g;
    push @parts, "the rule generating " . join(", ", @gen) . " reads it: " . join(", ", map { label($_) } @g);
  }
  for my $o (grep { /\.o$/ && reads($_, $p, $d, $b) } @stub_targets) {
    my @libs = sort keys %{$obj_libs{$o} // {}};
    my @o = obj_owners($o);
    $hit{$_} = $whole_hit{$_} = 1 for @o;
    hit_reason($_, $p, "$p (compiled into the stub object $o)") for @o;
    push @parts, "compiled into $o, " . (@libs ? "bundled by " . join(", ", map { label($_) } @libs)
      : "which no stub archive bundles, so every executable and test stanza of its directory (" . scalar(@o) . ")");
  }
  # An included fragment counts as the dune file of every directory including it.
  my @frag_of = @{$fragment_dirs{$p} // []};
  if ($covered{$p}) {
    push @parts, "compared with the old revision (above)";
  } elsif ($b eq "dune" || $b eq "dune-project" || $b eq "dune-workspace" || @frag_of) {
    for my $root (@frag_of ? @frag_of : ($d)) {
      $data_dirs{$root} = 1;
      $data_why{$root} //= $p;
      # A `dune` file's settings (`env`, `(dirs)`, ...) apply beneath it as well,
      # as `dune-project`'s language and extensions do: every directory under it.
      for (grep { under($_, $root) } $root, keys %$aliases) { $dune_dirs{$_} = 1; $dune_why{$_} //= $p }
      my @here = grep { under($st[$_]{dir}, $root) } 0 .. $#st;
      hit_reason($_, $p, "$p (a dune file over its directory, not compared across revisions)") for @here;
      push @ix, @here;
      # Its rules make the files there: a rule elsewhere reading one of them (a
      # fixture's generated sources) reads the change.
      my $n_made = 0;
      for my $a (grep { !under($_->{dir}, $root) } @al) {
        for my $t (@{$a->{own}}) {
          next unless grep({ under((dir_of($_))[0], $root) } keys %{$reads{$t}}) || grep { under($_->[0], $root) } @{$globs{$t}};
          $n_made++ unless $read_by{$p}{$a->{name}}++;
          $read_via{$p}{$a->{name}} //= "$t (it reads files the rules under $root make)";
          last;
        }
      }
      push @parts, "every per-test alias and all " . scalar(@here) . " stanza(s) " . ($root eq "." ? "of the project" : "under $root")
        . ($n_made ? "; files its rules make are read by the rules of $n_made alias(es) elsewhere" : "");
    }
  } elsif ($b =~ /^([^.]+)(?:\.[^.]+)*\.mli?$/) {
    my $mod = ucfirst $1;
    my $how = "module $mod of";
    @ix = @{$by_modkey{"$d\0$mod"} // []};
    $by_module = $mod if @ix;
    unless (@ix) {
      my @tests = grep { $st[$_]{kind} eq "tests" && $st[$_]{dir} eq $d } 0 .. $#st;
      my @named = grep { $st[$_]{units}{lcfirst $mod} } @tests;
      @ix = @named ? @named : @tests;
      $how = @named ? "test module $mod of" : "a module of a tests stanza here (dune does not describe theirs), so all of";
    }
    if (my @disabled = @ix || @gen ? () : @{disabled_stanzas($d)}) {
      push @parts, "stanzas disabled on system $system: " . join(", ", map {
        join(" ", $_->{kind}, dnames($_))
      } @disabled);
    } elsif (!@ix && !@gen) {
      # No stanza of this build owns it. One that dune lists but does not
      # compile here (an optional backend) may: those are reached, unknown
      # modules and all, and the path stays UNMAPPED either way.
      @ix = grep { $st[$_]{undescribed} && $st[$_]{dir} eq $d } 0 .. $#st;
      $how = "no stanza of this build owns $mod; reaching the stanzas dune lists here but does not build:";
      # No stanza lists it at all, compiled here or not, and a harness reads it
      # (a scan table's stub): mapped to those legs, named below.
      unless (!@ix && $harness_read{$p}) {
        $unmapped = 1;
        $unmapped_mod{$p} = [$d, $mod];
      }
    }
    push @parts, "$how " . (@ix > 3 ? scalar(@ix) . " stanzas" : join(", ", map { label($_) } @ix)) if @ix;
  }
  $hit{$_} = 1 for @ix;
  unless (@frag_of || $b =~ /^dune(?:-project|-workspace)?$/) {
    hit_reason($_, $p, "$p (" . (defined $by_module ? "its module $by_module" : $unmapped ? "a module no stanza of this build owns"
      : "a module of a tests stanza of its directory") . ")") for @ix;
  }
  if (defined $by_module) { $mod_hit{$_}{$by_module} = 1 for @ix; $mod_paths{$by_module}{$p} = 1 } else { $whole_hit{$_} = 1 for @ix }
  push @parts, "read by the rules of $n_read alias(es)" if $n_read;
  push @parts, "read by $harness_read{$p} harness(es), named below" if $harness_read{$p};
  print STDERR "affected: $p: " . join("; ", @parts) . "\n" if @parts;
  # A changed script an alias's rule names by its path (gh-ocannl-2304): the header's "Literal
  # dependencies". When that alias also runs sweep_harness_driver.exe, emit the sweep-driver reach
  # line for the scoped Linux shell leg (gh-ocannl-2414). When the Dune probe did not run, targetless
  # alias rules are opaque, so report sweep-driver reach unavailable rather than empty.
  if ($b =~ /\.sh\z/) {
    if (!$probe_ok) {
      print STDERR "affected: $p: sweep-driver reach unavailable (dune probe did not run)\n";
    }
    my %by_name = map { $_->{name} => $_ } grep { names_literally($_, $p) } @al;
    for my $n (sort keys %by_name) {
      print STDERR "affected: $p: named as a dependency by the rule of $n\n";
      print STDERR "affected: $p: reached by sweep driver alias $n\n" if runs_sweep_driver($by_name{$n});
    }
  }
  # Any path but a module source a stanza owns may be an input of an alias
  # action dune does not dump (see `opaque` below).
  unless ($b =~ /\.mli?$/ && @ix && !$unmapped) { for ($d, @{$copies{$p} // []}) { $data_dirs{$_} = 1; $data_why{$_} //= $p } }
  # A path no compiler or dune reads as a build input (a script, a config, a
  # golden, a note) may be any opaque action's `(deps)`, wherever it lies.
  unless ($b =~ /\.(?:mli?|mll|mly|c|h|cc|cpp|cxx|hh|hpp)$/ || $b =~ /^dune(?:-project|-workspace)?$/) { $any_data = 1; $any_data_why //= $p }
  if ($unmapped) { push @unmapped, $p } elsif (!@parts) { push @unread, $p }
}
# The files whose rules a compared dune edit changed (above), as the build has
# them: what reads one is reached as by a changed path -- an alias's own rules,
# a generated module's rule, a stub object's, a copy elsewhere -- and so is a
# stanza whose link or preprocessing rule reads one (`link_deps`,
# `preprocessor_deps`), which those rules are asked for here.
if (%delta_paths) {
  my %stanza_rule;   # .exe or .pp.ml(i) target -> stanza indices
  for my $d (sort keys %$targets) {
    for my $t (@{$targets->{$d}}) {
      my @ix = $t =~ /^(.+)\.(?:exe|bc)$/ ? @{$exe{"$d\0$1"} // []}
             : $t =~ /^([^.]+)\.pp\.mli?$/ ? @{$by_modkey{"$d\0" . ucfirst $1} // []} : ();
      $stanza_rule{in_dir($d, $_)} = [@ix] for @ix ? ($t) : ();
    }
  }
  # One dune cannot print (a rewriter it cannot drive, say) is one dune cannot
  # build either: its stanza has nothing to run, which is said.
  {
    local $rules_soft = 1;
    my @todo = grep { !$queried{$_} } sort keys %stanza_rule;
    unless (query_rules(@todo)) {
      for my $t (@todo) {
        next if $queried{$t} || query_rules($t);
        print STDERR "affected: dune rules $t failed (dune cannot build it either), so what it reads is not known\n";
      }
    }
  }
  fold($_) for sort keys %queried;
  for my $p (sort keys %delta_paths) {
    my ($d, $b) = dir_of($p);
    for my $a (@al) {
      for my $t (@{$a->{own}}) {
        next unless $t eq $p || reads($t, $p, $d, $b);
        $read_by{$p}{$a->{name}} = 1;
        $read_via{$p}{$a->{name}} //= $t eq $p ? "$t (its rule changed)" : $t;
        last;
      }
    }
    my $why = "$p (a target whose rule a compared dune edit changed)";
    for my $g (grep { reads($_, $p, $d, $b) } sort keys %generated) {
      for (@{$generated{$g}}) { $hit{$_} = $whole_hit{$_} = 1; hit_reason($_, $delta_paths{$p}, $why) }
    }
    for my $o (grep { /\.o$/ && reads($_, $p, $d, $b) } @stub_targets) {
      for (obj_owners($o)) { $hit{$_} = $whole_hit{$_} = 1; hit_reason($_, $delta_paths{$p}, $why) }
    }
    # A stanza's own module sources are its modules, not inputs of this kind.
    for my $t (grep { reads($_, $p, $d, $b) } sort keys %stanza_rule) {
      next if $b =~ /\.mli?$/ && grep { $_ eq $p } map { @$_ } values %{$st[$stanza_rule{$t}[0]]{mods} // {}};
      for (@{$stanza_rule{$t}}) { $hit{$_} = $whole_hit{$_} = 1; hit_reason($_, $delta_paths{$p}, $why) }
    }
    for (@{$copies{$p} // []}) { $data_dirs{$_} = 1; $data_why{$_} //= $p }
  }
}

# Reach, to a fixpoint: a stanza requiring a reached library is reached.
my %reached = %hit;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $i (0 .. $#st) {
    next if $reached{$i};
    if (grep { $reached{$_} } @{$st[$i]{to}}) { $reached{$i} = 1; $grew = 1 }
  }
}

# The link-only ranking (gh-ocannl-1431). Link reach is sound and blunt: an
# edit to one module of arrayjit reaches every test linking arrayjit, ~600
# aliases. A reached alias is ranked "link-only" when no reached owner of it
# links a library hit as a whole (by a stub, a generated module, a stanza
# edit: no name says what changed there), and none names the changed part of
# a changed module in its own code. Its own code: its stanza's sources,
# the sources of a project ppx rewriter it is preprocessed with, and, of the
# modules of the private (unpublished) libraries it links that that code
# names, the top-level definitions it names, comments left out (def_chunks: a
# definition by its name, a type by its name, constructors or fields, an
# `open`, `include` or effect always), to a fixpoint (gh-ocannl-1651). Names
# the changed part: mentions the module, or a name re-exporting it (`module A
# = ...M` with A not M, or an `include ...M`) qualified by, or with an open of,
# the module making that re-export (A is a name of it, not a global one), and, when the git
# hunks place every changed line in a top-level definition, one of those
# definitions too (see %path_defs). Mentions are tokens, the owner's own
# comments and strings included, so a name only ever errs towards "names it".
# THE BOUNDARY, stated: this reads names, never scopes -- a token is a mention
# wherever it stands, and a helper definition counts by its name alone, not by
# whether the call resolves to it; it never sees what the owner reaches through
# other library code (every compile goes through Low_level.optimize_proc, which
# no test names; gh-ocannl-1235's Register_tile miss was of this kind), nor
# through a public library's code, nor through what a project ppx's output
# names (its sources count whole, so ppx_cd's `Ir.Indexing` makes every %cd
# user name Indexing). So link-only aliases stay printed, LAST, and named on
# stderr; `--referenced-only` leaves them off stdout.
# Beside the tokens, the keys a re-exported name is read through
# (gh-ocannl-1765), each made within one file, since an `open` in a helper
# opens nothing in the test calling it: `X.N` for each qualified step of a
# path (`Ocannl.Context.Tn` gives `Ocannl.Context` and `Context.Tn`); `open X:
# N` for each capitalized token N of a file that may use X's names bare --
# `open X`/`open! X`, `include X`, `let open X in`, a local open `X.(...)` (or
# `.[`, `.{`), each by the path's last module, and X's own source; and `alias M
# X: N` for `M.N` in a file that binds `module M = ...X`. No key can collide
# with an identifier token. UNIT is the text the keys are for (a definition of
# a helper module), SCOPE the whole file its opens and aliases are read from.
sub file_scope {   # SCOPE FILE -> [{X opened => 1}, {M => {X => 1}}]
  my ($scope, $f) = @_;
  my (%open, %alias);
  $open{$1} = 1 while $scope =~ /\b([A-Z][\w']*)\s*\.\s*[(\[{]/g;
  while ($scope =~ /\b(?:open!?|include|module\s+([A-Z][\w']*)\s*=)\s*((?:[A-Z][\w']*\s*\.\s*)*[A-Z][\w']*)/g) {
    my ($m, $path) = ($1, $2);
    my ($x) = $path =~ /([A-Z][\w']*)\z/;
    if (defined $m) { $alias{$m}{$x} = 1 } else { $open{$x} = 1 }
  }
  # An open of a local alias opens what it aliases (local review), to a fixpoint.
  for (my $grew = 1; $grew;) {
    $grew = 0;
    for my $m (keys %open) { for my $x (keys %{$alias{$m} // {}}) { $grew = $open{$x} = 1 unless $open{$x} } }
  }
  $open{ucfirst $1} = 1 if $f =~ m{([A-Za-z_][\w']*)\.mli?\z};
  return [\%open, \%alias];
}
sub qual_keys {   # UNIT [OPENS ALIASES] (file_scope) -> (KEY...)
  my ($unit, $sc) = @_;
  my ($open, $alias) = @$sc;
  my (@k, %cap);
  while ($unit =~ /\b([A-Z][\w']*)\s*\.\s*(?=([A-Za-z_][\w']*))/g) {
    my ($m, $n) = ($1, $2);
    push @k, "$m.$n";
    push @k, map { "alias $m $_: $n" } keys %{$alias->{$m} // {}};
  }
  $cap{$1} = 1 while $unit =~ /\b([A-Z][\w']*)/g;
  for my $x (keys %$open) { push @k, map { "open $x: $_" } keys %cap }
  return @k;
}
my %tok_cache;
sub tokens_of {
  my ($f) = @_;
  return $tok_cache{$f} //= do {
    my %t;
    if (open my $fh, "<", $f) {
      local $/;
      my $s = <$fh> // "";
      $t{$1} = 1 while $s =~ /\b([A-Za-z_][A-Za-z0-9_']*)/g;
      $t{$_} = 1 for qual_keys($s, file_scope($s, $f));
    }
    \%t;
  };
}
# A private helper module's top-level definitions, each with the names it
# answers to and the tokens of its code (gh-ocannl-1651): a helper library such
# as test/support's ll_test names Autotune in a few of its ~90 definitions, and
# every test linking it named Autotune through all of them. The code is read
# with its comments blanked (code_of), so a doc comment naming a module is no
# mention of it. A definition starts at a column-0 `let`, `val`, `external`,
# `module`, `class`, `exception` or `type` naming one identifier, and runs to
# the next column-0 line that is not `and` (an `and` adds its name). A `type`
# answers to its name, its constructors and its fields. Everything else at
# column 0 -- `open`, `include`, `let () =`, a pattern or operator binding, an
# attribute -- counts always, as do a definition's own lines whatever names
# they mention. A misread line boundary only merges two definitions, which errs
# towards "names it".
sub code_of {   # OCaml text with every comment blanked, newlines kept; strings inside comments are comment
  my ($s, $mark) = @_;   # MARK: every line a literal spans is made non-blank (`S`), its newlines kept
  my ($out, $depth) = ("", 0);
  pos($s) = 0;
  while (pos($s) < length $s) {
    my $piece;
    if ($s =~ /\G\(\*/gc) { $depth++; $piece = "  " }
    elsif ($depth && $s =~ /\G\*\)/gc) { $depth--; $piece = "  " }
    elsif ($s =~ /\G("(?:[^"\\]++|\\.)*+")/gcs || $s =~ /\G(\{([a-z_]*)\|.*?\|\2\})/gcs
           || $s =~ /\G('(?:[^'\\\n]|\\(?:[\\'"ntbr ]|[0-9]{3}|x[0-9A-Fa-f]{2}|o[0-7]{3}))')/gc
           || $s =~ /\G([^(*"{'\n]+|.)/gcs) {
      $piece = $1;
      $piece = $piece =~ s/[^\n]/S/gr =~ s/\n/\nS/gr if $mark && !$depth && $piece =~ /^["'{]./s;
    }
    $out .= $depth || $piece eq "  " ? $piece =~ s/[^\n]/ /gr : $piece;
  }
  return $out;
}
my %chunk_cache;
sub def_chunks {   # FILE -> [{ names => [NAME...] or undef (always counted), tok => {TOKEN => 1} }]
  my ($f) = @_;
  return $chunk_cache{$f} //= do {
    my $s = "";
    if (open my $fh, "<", $f) { local $/; $s = <$fh> // "" }
    my (@chunks, $cur);
    my $code = code_of($s);
    my $scope = file_scope($code, $f);
    for my $l (split /\n/, $code) {
      if ($l =~ /^and\b/ && $cur) {
        if ($cur->{names} && $l =~ /^and\s+(?:type\s+)?(?:'[\w']+\s+|\([^()]*\)\s+)?([A-Za-z_][\w']*)\b(?!\s*,)/) {
          push @{$cur->{names}}, $1;
        } else { $cur->{names} = undef }
        push @{$cur->{lines}}, $l;
        next;
      }
      if ($l =~ /^[^\s)\]}|]/ && $l !~ /^(?:end|done|in|then|else|with)\b/) {
        push @chunks, $cur if $cur;
        my ($name, $type);
        $name = $1 if $l =~ /^let(?:%\w+)?\s+(?:rec\s+)?([a-z_][\w']*)\b(?!\s*,)/ && $1 ne "_";
        $name = $1 if $l =~ /^(?:val|external)\s+([a-z_][\w']*)/;
        $name = $1 if $l =~ /^(?:module|class|exception)\s+(?:rec\s+|type\s+)?([A-Za-z_][\w']*)/;
        ($name, $type) = ($1, 1) if $l =~ /^type\s+(?:nonrec\s+)?(?:'[\w']+\s+|\([^()]*\)\s+)?([a-z_][\w']*)/;
        $cur = { names => defined $name ? [$name] : undef, lines => [$l], type => $type };
        next;
      }
      # A definition's own lines (indented, blank, or a closing `end`, `)`, `|`, ...).
      if ($cur) { push @{$cur->{lines}}, $l } else { $cur = { names => undef, lines => [$l] } }
    }
    push @chunks, $cur if $cur;
    for my $c (@chunks) {
      my %t;
      my $text = join("\n", @{$c->{lines}});
      $t{$1} = 1 while $text =~ /\b([A-Za-z_][A-Za-z0-9_']*)/g;
      $t{$_} = 1 for qual_keys($text, $scope);
      # A type answers to its constructors and fields too: what users name. Read
      # loosely (every capitalized word after `|`, `=`, `[` or `of`, every word
      # before a `:`), which errs towards more names.
      if ($c->{type} && $c->{names}) {
        push @{$c->{names}}, $text =~ /(?:[|=\[]|\bof\b)\s*`?([A-Z][\w']*)/g;
        push @{$c->{names}}, $text =~ /\b([a-z_][\w']*)\s*:(?!:)/g;
      }
      $c = { names => $c->{names}, tok => \%t };
    }
    \@chunks;
  };
}
sub dnames { my ($s) = @_; map { ref $_ ? () : $_ } map { @{dfield($s->{x}, $_) // []} } qw(name names public_name public_names) }
sub static_of {   # the dune stanza an @st entry was read from, or undef
  my ($i) = @_;
  my $s = $st[$i];
  return $s->{static} if exists $s->{static};
  my %n = map { $_ => 1 } @{$s->{names}};
  my $lib = $s->{kind} eq "library";
  my @m = grep {
    $_->{dir} eq $s->{dir} && $_->{kind} =~ /^(?:library|executables?|tests?)$/
      && ($_->{kind} eq "library") == $lib && grep { $n{$_} } dnames($_)
  } @dstanzas;
  return $s->{static} = @m == 1 ? $m[0] : undef;
}
sub sources_unknown { my ($i, $why) = @_; $st[$i]{unknown_why} //= $why; return undef }
sub stanza_sources {   # an @st entry's source files, or undef when they are not known
  my ($i) = @_;
  my $s = $st[$i];
  # Each unknown says why, for `--why` (gh-ocannl-1711).
  return sources_unknown($i, "it has a generated module (a (select), a rule's output)") if $s->{gen_mod};   # unknown, not empty
  return [map { @$_ } values %{$s->{mods}}] if $s->{mods};
  my $ds = static_of($i) // return sources_unknown($i, "its dune stanza is not read here");
  # Under `(include_subdirs ...)` its sources extend into subdirectories: unknown.
  return sources_unknown($i, "(include_subdirs) extends its sources beneath its directory")
    if grep { $_->{kind} eq "include_subdirs" && under($s->{dir}, $_->{dir}) } @dstanzas;
  my $mods = dfield($ds->{x}, "modules");
  my @names;
  if ($mods) {
    return sources_unknown($i, "its (modules) field is not a literal list")
      if grep { ref $_ || !/^[A-Za-z_][A-Za-z0-9_']*$/ } @$mods;   # :standard, \, (:include ...)
    @names = map { lcfirst } @$mods;
  } else {   # every module of the directory no other stanza there claims
    my %claimed;
    for my $o (grep { $_ != $ds && $_->{dir} eq $s->{dir} } @dstanzas) {
      my $om = dfield($o->{x}, "modules") // next;
      $claimed{lcfirst $_} = 1 for grep { !ref } @$om;
    }
    opendir my $dh, $s->{dir} or return sources_unknown($i, "its directory cannot be listed");
    @names = grep { !$claimed{$_} } map { /^([A-Za-z_][A-Za-z0-9_']*)\.mli?$/ ? lcfirst $1 : () } readdir $dh;
  }
  my @files;
  for my $n (@names) {
    my @f = grep { -f } map { in_dir($s->{dir}, "$n.$_") } "ml", "mli";
    # A `(select)` module: every arm's file, whichever this build chose.
    unless (@f) {
      for my $ext ("ml", "mli") {
        my $arms = select_arms($ds, "$n.$ext") // next;
        return sources_unknown($i, "its module " . ucfirst($n) . " is a (select) with an arm not read here")
          if !@$arms || grep { !-f } @$arms;
        push @f, @$arms;
      }
    }
    @f or return sources_unknown($i, "its module " . ucfirst($n) . " has no source here (generated)");
    push @files, @f;
  }
  return \@files;
}
sub select_arms {   # a dune stanza's `(select FILE from (... -> ARM)...)` -> [each ARM's path], or undef: no such select
  my ($ds, $file) = @_;
  for my $l (@{dfield($ds->{x}, "libraries") // []}) {
    next unless ref $l && @$l > 1 && !ref $l->[0] && $l->[0] eq "select" && !ref $l->[1] && $l->[1] eq $file;
    my @arms;
    for my $arm (@$l[3 .. $#$l]) {
      return [] unless ref $arm && @$arm >= 2 && !ref $arm->[-1] && !ref $arm->[-2] && $arm->[-2] eq "->";
      push @arms, in_dir($ds->{dir}, $arm->[-1]);
    }
    return \@arms;
  }
  return undef;
}
sub is_private { my $ds = static_of($_[0]); $ds && !grep { defined dfield($ds->{x}, $_) } qw(public_name public_names) }
sub flag_atoms {   # every atom of some flags fields, undef for a part not read as atoms
  map { !defined $_ ? () : ref $_ ? (@$_ && !ref $_->[0] && $_->[0] =~ /^:include$/ ? (undef) : flag_atoms(@$_)) : $_ } @_;
}
sub own_tokens {   # the names an owner's own code mentions, or undef when its code is not known
  my ($i) = @_;
  return $st[$i]{own_tokens} if exists $st[$i]{own_tokens};
  my $files = stanza_sources($i);
  my %tok;
  # Where each token was first found, for `--why` (gh-ocannl-1711): [FILE, KIND]
  # for the owner's own code, or for a definition of a private library it links [FILE, "definition", [ITS NAMES] or undef, THE
  # NAME THE OWNER'S CODE NAMES IT BY or undef, ITS MODULE] (undef names: a
  # top-level form counted always, once the module is named).
  my %src;
  my $why_not = sub { $st[$i]{unknown_why} //= $_[0]; $files = undef };
  if ($files) {
    my %kind = map { $_ => "its own source" } @$files;
    my @more = @$files;
    if (my $ds = static_of($i)) {
      my $pp = dfield($ds->{x}, "preprocess");
      if ($pp && @$pp) {
        # (pps NAME... -- flags): the project rewriters' own sources count.
        if (ref $pp->[0] && ($pp->[0][0] // "") eq "pps" && @$pp == 1) {
          for my $n (map { ref $_ ? () : $_ } @{$pp->[0]}[1 .. $#{$pp->[0]}]) {
            last if $n eq "--";
            next unless defined $lib_by_name{$n};
            # A rewriter that changed (or links what changed) rewrites this code
            # under no name it shows: unknown.
            $why_not->("its ppx rewriter $n is reached (changed, or links what changed)") if $reached{$lib_by_name{$n}};
            my $pp_files = stanza_sources($lib_by_name{$n}) // ($why_not->("its ppx rewriter ${n}'s sources are not all known"), []);
            $kind{$_} //= "a source of its ppx rewriter $n" for @$pp_files;   # `--why`
            push @more, @$pp_files;
          }
        } elsif (!(!ref $pp->[0] && $pp->[0] eq "no_preprocessing")) {   # per_module, action: not read
          $why_not->("its (preprocess) is not a plain (pps ...)");
        }
      }
      # `-open M` in its flags, or in an `env` stanza above it, lets its code use M's
      # names bare: unknown.
      # Flags with an `(:include ...)` or any other non-literal element are not
      # known either.
      $why_not->("its flags hold -open or a part not read as literal atoms")
        if grep { !defined || $_ eq "-open" || /^:(?!standard$)/ || /%\{/ } flag_atoms(map { dfield($ds->{x}, $_) } qw(flags ocamlc_flags ocamlopt_flags));
      $why_not->("an env stanza above it passes -open")
        if grep { $_->{kind} eq "env" && under($ds->{dir}, $_->{dir}) && canon($_->{x}) =~ /"-open"/ } @dstanzas;
    }
    if ($files) {
      for my $f (@more) { for (sort keys %{tokens_of($f)}) { $tok{$_} = 1; $src{$_} //= [$f, $kind{$f}] } }
      my (%priv, %seen);
      my @q = @{$st[$i]{to}};
      while (defined(my $j = shift @q)) {
        # A changed library is not the owner's own code, even when private: its modules
        # name their own definitions.
        next if $seen{$j}++ || $st[$j]{kind} ne "library" || !is_private($j) || $mod_hit{$j};
        # The same uncertainty checks as the owner's own stanza: a helper whose code
        # is not all known leaves the owner's unknown too.
        defined own_tokens($j) or do {
          $st[$i]{unknown_why} //= "the private " . label($j) . " it links holds code not all known: $st[$j]{unknown_why}";
          return $st[$i]{own_tokens} = undef;
        };
        $priv{$j} = 1;
        push @q, @{$st[$j]{to}};
      }
      # A helper module the code names counts by the definitions it names
      # (def_chunks), to a fixpoint: a named definition's own mentions name more.
      my %done;
      for (my $grew = 1; $grew;) {
        $grew = 0;
        for my $j (sort keys %priv) {
          for my $m (sort keys %{$st[$j]{mods} // {}}) {
            next unless $tok{$m};
            for my $f (@{$st[$j]{mods}{$m}}) {
              my $chunks = def_chunks($f);
              for my $k (0 .. $#$chunks) {
                my $c = $chunks->[$k];
                my @by;
                next if $done{"$f\0$k"} || ($c->{names} && !(@by = grep { $tok{$_} } @{$c->{names}}));
                $done{"$f\0$k"} = $grew = 1;
                for (sort keys %{$c->{tok}}) { $tok{$_} = 1; $src{$_} //= [$f, "definition", $c->{names}, $by[0], $m] }
              }
            }
          }
        }
      }
    }
  }
  $st[$i]{tok_src} = \%src;
  return $st[$i]{own_tokens} = $files ? \%tok : undef;
}
# Which top-level definitions a changed `.ml`/`.mli` touched, from the git
# hunks and both sides of the file: each changed line belongs to the nearest
# definition starting at column 0 above it (`let`, `and`, `type`, `val`,
# `module`, `external`, `exception`, `class`), on the side the line is on. A
# line in a `type` touches that type by its name, the constructors and fields
# the line itself names, and the functions a deriver makes from the type's
# name (gh-ocannl-1651: adding a field or a constructor is what users of the
# others do not name); an attribute line there (`[@@deriving ...]`, `[@...]`)
# touches the module as a whole. A line under `open`, `include`, an `and`, a
# `let () =`/`let _ =` effect or a pattern binding, or above every definition,
# touches the module as a whole; so does any file without hunks (`--files`, a
# new file).
my %path_defs;   # path -> {NAME => 1}, or undef: the whole module
sub type_names {   # TYPE-NAME CHANGED-LINE -> the names a change to that line of the type touches
  my ($t, $line) = @_;
  my $code = code_of($line);
  my @n = ($t, $code =~ /(?:^|[|=\[]|\bof\b)\s*`?([A-Z][\w']*)/g, $code =~ /\b([a-z_][\w']*)\s*:(?!:)/g);
  # What ppx_compare, ppx_hash, ppx_sexp_conv, ppx_enumerate, ppx_fields_conv and
  # ppx_variants_conv make of it.
  push @n, $t eq "t" ? qw(compare equal hash hash_fold_t sexp_of_t t_of_sexp all Fields Variants)
                     : map { s/N/$t/r } qw(compare_N equal_N hash_N hash_fold_N sexp_of_N N_of_sexp all_of_N Fields_of_N Variants_of_N);
  return [grep { !/^(?:of|mutable|private|nonrec)$/ } @n];
}
sub enclosing_def {   # -> [NAME...], or undef: the whole module
  my ($lines, $n) = @_;   # 1-based line number
  for (my $k = $n; $k >= 1; $k--) {
    my $l = $lines->[$k - 1] // next;
    next if $l =~ /^\s/ || $l eq "" || $l =~ /^(?:\(\*|\[\@|\)|;;|\||end\b|\})/;
    # Whitelisted forms only: a `let` value, a `val`/`external`, a type, a module,
    # class or exception name. An `and` (of a type or of a value) touches the
    # whole module.
    if ($l =~ /^type\s+(?:nonrec\s+)?(?:'[\w']+\s+|\([^()]*\)\s+)?([a-z_][\w']*)/) {
      my $t = $1;
      my $changed = $lines->[$n - 1] // "";
      return undef if $changed =~ /\[\@/;
      return type_names($t, $changed);
    }
    return undef if $l =~ /^(?:type|and)\b/;
    return [$1] if $l =~ /^let\s+(?:rec\s+)?(?:%\w+\s+)?([a-z_][\w']*)\b(?!\s*,)/ && $1 ne "_";
    return [$1] if $l =~ /^let\s+(?:rec\s+)?\(\s*([^\s()]+)\s*\)/;   # an operator
    return [$1] if $l =~ /^(?:val|external)\s+([a-z_][\w']*)/ || $l =~ /^val\s+\(\s*([^\s()]+)\s*\)/;
    return [$1] if $l =~ /^(?:module|class|exception)\s+(?:rec\s+|type\s+)?([A-Za-z_][\w']*)/;
    return undef;   # open, include, let () =, a pattern: the whole module
  }
  return undef;
}
if (-f "$tmp/ml-diff") {
  my (%hunks, $cur, $gone);
  for (split /\n/, slurp("$tmp/ml-diff")) {
    if (/^diff /) { $cur = $gone = undef; next }
    if (/^--- (?:a\/(.*)|\/dev\/null)$/) { $gone = $1; next }
    if (/^\+\+\+ (?:b\/(.*)|\/dev\/null)$/) { $cur = $1 // $gone; next }
    if (/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/ && defined $cur) {
      push @{$hunks{$cur}}, [old => $1, $2 // 1], [new => $3, $4 // 1];
    }
  }
  for my $p (keys %hunks) {
    my %defs;
    my %side = map { my $f = "$tmp/$_/$p"; ($_ => [-f $f ? split(/\n/, slurp($f), -1) : ()]) } qw(old new);
    # A changed line holding no code -- blank, or comment only, outside every
    # literal -- changes no definition (gh-ocannl-1765: the blank line above an
    # added definition was attributed to the definition above it).
    my %code = map { my $f = "$tmp/$_/$p"; ($_ => [-f $f ? split(/\n/, code_of(slurp($f), 1), -1) : ()]) } qw(old new);
    my %inert = map { my $l = $code{$_}; ($_ => { map { $l->[$_] =~ /\S/ ? () : ($_ + 1 => 1) } 0 .. $#$l }) } qw(old new);
    # A file new on this side (an `.mli` added beside its `.ml`, say) or gone from
    # it can hide or expose any member: the whole module.
    if (!-f "$tmp/old/$p" || !-f "$tmp/new/$p") { $path_defs{$p} = undef; next }
    # A delimiter on a changed line can comment out, or bring back, lines git
    # shows unchanged (a definition between an added `(*` and `*)`), and a quote
    # can turn them into a literal: every unchanged line must read as the same
    # code on both sides, or the whole module changed (local review).
    my ($o, $n, $moved) = (1, 1, 0);
    my @hs = @{$hunks{$p}};
    for (my $k = 0; $k <= $#hs; $k += 2) {
      my ((undef, $os, $oc), (undef, $ns, $nc)) = (@{$hs[$k]}, @{$hs[$k + 1]});
      my $before = $oc ? $os : $os + 1;   # the first old line of the hunk, or the one after an insertion
      for (; $o < $before; $o++, $n++) { $moved = 1 if ($code{old}[$o - 1] // "") ne ($code{new}[$n - 1] // "") }
      $o += $oc;
      $n += $nc;
    }
    for (; $o <= @{$code{old}}; $o++, $n++) { $moved = 1 if ($code{old}[$o - 1] // "") ne ($code{new}[$n - 1] // "") }
    if ($moved) { $path_defs{$p} = undef; next }
    for my $h (@{$hunks{$p}}) {
      my ($which, $start, $count) = @$h;
      next unless $count;
      @{$side{$which}} or do { %defs = ("\0whole" => 1); last };
      for my $n ($start .. $start + $count - 1) {
        next if $inert{$which}{$n};
        my $names = enclosing_def($side{$which}, $n);
        if (!defined $names) { %defs = (); $defs{"\0whole"} = 1; last }
        # An operator or other non-identifier: tokens hold identifiers only, so the
        # whole module counts.
        if (grep { !/^[A-Za-z_][\w']*$/ } @$names) { %defs = ("\0whole" => 1); last }
        $defs{$_} = 1 for @$names;
      }
      last if $defs{"\0whole"};
    }
    # No definition at all: only comments or blank lines changed.
    $path_defs{$p} = $defs{"\0whole"} ? undef : \%defs;
  }
}
# The names a changed module answers to: its own, and those re-exporting it,
# each with the modules it is visible through (below).
# A stanza reached from one hit as a whole is never link-only: no name says
# what changed there.
# So is one whose generated module or stub a reached generator writes: what the
# generator changed is code the stanza compiles, under no name it would show.
my %reached_whole = %whole_hit;
$reached_whole{$_} = 1 for grep { my $i = $_; grep { $reached{$_} } @{$st[$i]{gen_to} // []} } 0 .. $#st;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $i (0 .. $#st) {
    next if $reached_whole{$i} || !grep { $reached_whole{$_} } @{$st[$i]{to}};
    $reached_whole{$i} = $grew = 1;
  }
}
my $rank_links = 1;
my %changed_names;   # NAME -> {changed module it answers for => undef, or {module NAME is visible through => 1}}
my %mod_defs;        # changed module -> {definition => 1}, or undef: the whole module
if ($rank_links) {
  for my $m (map { keys %{$mod_hit{$_}} } grep { $st[$_]{kind} eq "library" } keys %mod_hit) {
    $changed_names{$m}{$m} = undef;
    my @p = keys %{$mod_paths{$m}};
    $mod_defs{$m} = (grep { !$path_defs{$_} } @p) ? undef : { map { %{$path_defs{$_}} } @p };
  }
  $rank_links = 0 unless %changed_names;
}
if ($rank_links) {
  # M -> {name some library module re-exports M under => {that module => 1}}. A
  # re-export is a name of the module that makes it: `module Lazy = Utils.Lazy`
  # in hip_backend.ml is `Hip_backend.Lazy`, and code naming `Lazy` without
  # naming Hip_backend means Stdlib's (gh-ocannl-1651: counted bare, it made
  # every test calling `Lazy.force` name Utils). So a re-exported name names M
  # only where the module re-exporting it qualifies it (`Context.Tn`), or is
  # opened, included or aliased, or in that module's own code (qual_keys): any
  # other token of the re-exporter is not enough (gh-ocannl-1765: `Tn` beside a
  # `Tensor.print` made four of a `tnode.ml` edit's owners referenced).
  my %as;
  for my $j (grep { $st[$_]{kind} eq "library" } 0 .. $#st) {
    for my $x (keys %{$st[$j]{mods} // {}}) {
      for my $f (@{$st[$j]{mods}{$x}}) {
        open my $fh, "<", $f or next;
        my $s = do { local $/; <$fh> // "" };
        # What a top-level `module NAME ... = PATH` or `include PATH` re-exports: the
        # module PATH names (its last component), read up to the first `struct`,
        # `sig`, `end`, `let`, `val`, blank or unindented line. A signature
        # constraint is read past (`module G : module type of M = M`, `include (M :
        # S)`). A functor application makes a new module whose code calls its
        # arguments: reached through other code, like any caller, so not read.
        while ($s =~ /^(module|include)\b/mg) {
          my ($kw, $at) = ($1, pos($s));
          my $head = substr($s, $at, 400);
          $head =~ s/(?:\b(?:struct|sig|end|let|val)\b|\n[ \t]*\n|\n(?=\S)).*\z//s;
          my ($as_name, $rhs);
          if ($kw eq "module") {
            next unless $head =~ /^\s*(?:rec\s+)?([A-Z][\w']*)/;
            $as_name = $1;
            ($rhs) = $head =~ /=([^=]*)\z/s or next;   # a declaration: no right-hand side
          } else { ($as_name, $rhs) = ($x, $head =~ s/:.*\z//sr) }
          $rhs =~ s/\(\*.*?\*\)//gs;   # comments
          my @names = $rhs =~ /^\s*\(?\s*([A-Z][\w']*)\s*(?:[:)]|\z)/s ? ($1) : ();
          # A right-hand side not read as one plain module name (a qualified path,
          # a functor application, anything else): uncertain, so every module name
          # in it counts as re-exported, and the wrapper's users are never
          # link-only on its account.
          @names = $rhs =~ /\b([A-Z][\w']*)/g if !@names && $rhs =~ /\S/;
          $as{$_}{$as_name}{$x} = 1 for grep { $_ ne $as_name } @names;
        }
      }
    }
  }
  # To a fixpoint: a re-export of a re-export is visible through the outer
  # re-exporter (whatever made the inner one visible is that library code's).
  my $size = sub { my $c = $changed_names{$_[0]} // {}; join ",", map { $_ . ":" . (defined $c->{$_} ? join("+", sort keys %{$c->{$_}}) : "") } sort keys %$c };
  my @q = keys %changed_names;
  while (defined(my $m = shift @q)) {
    for my $n (keys %{$as{$m} // {}}) {
      my $had = $size->($n);
      for my $orig (keys %{$changed_names{$m}}) {
        next if exists $changed_names{$n}{$orig} && !defined $changed_names{$n}{$orig};   # already unconditional
        $changed_names{$n}{$orig}{$_} = 1 for keys %{$as{$m}{$n}};
      }
      push @q, $n if $size->($n) ne $had;
    }
  }
}
# Where a token of an owner's code stands (`--why`, gh-ocannl-1711): its file,
# and for a helper definition the token that pulled that definition in, back
# to the owner's own code.
sub tok_where {
  my ($o, $t) = @_;
  my (@chain, %seen);
  while (my $s = ($st[$o]{tok_src} // {})->{$t}) {
    last if $seen{$t}++;
    my ($f, $kind, $names, $by, $m) = @$s;
    if ($kind ne "definition") { push @chain, "`$t` at $f ($kind)"; last }
    push @chain, "`$t` at $f (" . ($names ? "definition `$names->[0]`" : "a top-level form, counted whenever $m is named") . ")";
    $t = $by // $m;
  }
  return @chain ? join(" <- ", @chain) : "`$t`";
}
# The breadth-first chain from a reached stanza down its `to` edges (a library
# it requires, a generator its stub runs) to the nearest one TARGET accepts.
# Stanza index breaks ties between shortest routes for reproducible explanations.
sub reach_chain {   # INDEX TARGET -> (indices from INDEX to the target), or ()
  my ($o, $target) = @_;
  my %prev = ($o => -1);
  my @q = ($o);
  while (defined(my $i = shift @q)) {
    if ($target->($i)) {
      my @c;
      for (my $k = $i; $k != -1; $k = $prev{$k}) { unshift @c, $k }
      return @c;
    }
    for my $j (nsort(@{$st[$i]{to}})) { next if exists $prev{$j}; $prev{$j} = $i; push @q, $j }
  }
  return ();
}
sub chain_text { join(" -> ", map { label($_) } @_) }
# Whether a reached owner's code names a changed module's changed part, and
# why: (1, WHY, PATHS) when it does or ranks first for another reason;
# (0, WHY) when it is link-only on its own account. PATHS records any known
# changed paths that earned the first-tier rank. `link_only` reads the rank;
# `--why` prints the explanation and the cross-cutting summary counts paths.
sub read_owner_names {
  my ($o) = @_;
  return (1, "nothing is ranked: no module of a library changed by name (a stanza or a generated file did, or only data)")
    unless $rank_links;
  return (1, "it is itself reached by " . ($hit_why{$o} // "a changed path") . ", so no name says what changed there",
    [$hit_path{$o}]) if $hit{$o};
  if ($reached_whole{$o}) {
    my @c = reach_chain($o, sub { $whole_hit{$_[0]} || grep { $reached{$_} } @{$st[$_[0]]{gen_to} // []} });
    return (1, "it links what a changed path reached as a whole, under no name it would show: "
      . chain_text(@c) . ", which " . ($hit_why{$c[-1]} // "a changed path") . " reached",
      [$hit_path{$c[-1]}]) if @c && $whole_hit{$c[-1]};
    if (@c) {
      my ($g) = grep { $reached{$_} } @{$st[$c[-1]]{gen_to}};
      my @gc = reach_chain($g, sub { $hit{$_[0]} });
      return (1, "it compiles what a reached generator writes (a header, a module), under no name it would show: "
        . chain_text(@c, @gc) . ", which " . ($hit_why{$gc[-1] // $g} // "a changed path") . " reached",
        [$hit_path{$gc[-1] // $g}]);
    }
    return (1, "it links what a changed path reached as a whole (a stub, a generated module), under no name it would show");
  }
  my $tok = own_tokens($o) // return (1, "its code is not all known: " . ($st[$o]{unknown_why} // "a reason not recorded"));
  my @missed;
  my %aliases;   # X -> {the local aliases M of `module M = ...X` in this code => 1}
  for (keys %$tok) { $aliases{$2}{$1} = 1 if /^alias (\S+) (\S+):/ }
  my $through = sub {   # RE-EXPORTER NAME -> the key (qual_keys) reading NAME as RE-EXPORTER's, or undef
    my ($x, $n) = @_;
    # `include M` in X re-exports M's members as X's own: naming X is the reference.
    return $n if $x eq $n && $tok->{$n};
    my ($k) = grep { $tok->{$_} } "$x.$n", "open $x: $n", map { "alias $_ $x: $n" } sort keys %{$aliases{$x} // {}};
    return $k;
  };
  for my $m (sort keys %mod_defs) {
    my (@as, $why);
    for my $n (sort grep { $tok->{$_} && exists $changed_names{$_}{$m} } keys %changed_names) {
      my $via = $changed_names{$n}{$m};
      my @by = $via ? grep { $through->($_, $n) } sort keys %$via : ();
      next if $via && !@by;
      # A re-export is told by where it is read through its re-exporter.
      push @as, $n eq $m ? tok_where($o, $n)
        : "`$n`, re-exporting $m from " . join("/", @by) . ": " . join(" and ", map { tok_where($o, $through->($_, $n)) } @by);
    }
    my $defs = $mod_defs{$m};
    my @paths = sort grep { $_ ne "\0membership" } keys %{$mod_paths{$m}};
    if (@as && !$defs) {
      return (1, "it names $m: " . join("; ", @as) . "; and $m changed as a whole",
        [grep { !$path_defs{$_} } @paths]);
    }
    if (@as) {
      my @d = sort grep { $tok->{$_} } keys %$defs;
      if (@d) {
        my @witnesses = grep { my $p = $_; grep { $path_defs{$p}{$_} } @d } @paths;
        return (1, "it names $m: " . join("; ", @as) . "; and its changed definition(s): " . join("; ", map { tok_where($o, $_) } @d),
          \@witnesses);
      }
      push @missed, "names $m (" . join("; ", @as) . ") but " . (%$defs ? "none of its changed definitions (" . join(", ", sort keys %$defs) . ")"
        : "no definition of it changed (only comments or blank lines did)");
    } else {
      my @re = sort grep { $_ ne $m && exists $changed_names{$_}{$m} } keys %changed_names;
      push @missed, "names neither $m nor a re-export of it through the module making it"
        . (@re ? " (" . join(", ", map { my $v = $changed_names{$_}{$m}; "$_ through " . join("/", sort keys %$v) } @re) . ")" : "");
    }
  }
  return (0, "its code (its sources, its project ppx's, the helper definitions it names) " . join("; ", @missed));
}
my %owner_names;
sub owner_names { @{$owner_names{$_[0]} //= [read_owner_names($_[0])]} }
sub link_only {   # reached owners -> true when none of them names a changed module's changed part
  return 0 unless $rank_links;
  for my $o (@_) { return 0 if (owner_names($o))[0] }
  return 1;
}

# Backend-default marker routing (gh-ocannl-2382): one shared reading feeds
# tier promotion, `--why`, `cross_reason`, and `--gpu-gated` so all four agree.
my %declared_touch_cache;
sub declared_touch {   # reached owner index -> arrayref of matched default records
  my ($o) = @_;
  return $declared_touch_cache{$o} if exists $declared_touch_cache{$o};
  return $declared_touch_cache{$o} = [] unless $rank_links && @default_readers;
  my %names = map { $_ => 1 } @{$st[$o]{names}};
  my %read_defaults; # dname -> undef (unpinned) or hashref of reader-pinned backends
  for my $dr (@default_readers) {
    next unless $dr->{dir} eq $st[$o]{dir} && $dr->{kind} =~ /^(?:library|executables?|tests?)$/;
    next unless grep { $names{$_} } dnames($dr);
    for my $r (@{$dr->{reads}}) {
      my $dn = $r->[0];
      if (!exists $read_defaults{$dn}) {
        $read_defaults{$dn} = $dr->{stanza_bks} ? { %{$dr->{stanza_bks}} } : undef;
      } elsif (defined $read_defaults{$dn}) {
        if (!$dr->{stanza_bks}) { $read_defaults{$dn} = undef }
        else { $read_defaults{$dn}{$_} = 1 for keys %{$dr->{stanza_bks}} }
      }
    }
  }
  my @hits;
  for my $dname (sort keys %read_defaults) {
    my $def = $default_defs{$dname} // next;
    my $owner_bks = $read_defaults{$dname};
    my @eff_bks = $owner_bks ? (grep { $owner_bks->{$_} } @{$def->{backends}}) : @{$def->{backends}};
    next unless @eff_bks;
    my (@matched_defs, @matched_whole, %witness_paths);
    for my $tg (@{$def->{targets}}) {
      my ($m, $d) = @$tg;
      next unless exists $mod_defs{$m};
      # Scope target matching to the declaring library's own directory (including
      # any `(select ...)` arms), so a same-named module in another library cannot match.
      my $stem = lcfirst $m;
      my %allowed;
      for my $ext ("ml", "mli") {
        $allowed{in_dir($def->{dir}, "$stem.$ext")} = 1;
        my $arms = select_arms($def->{ds}, "$stem.$ext");
        $allowed{$_} = 1 for @{$arms // []};
      }
      my @paths = sort grep { $_ ne "\0membership" && $allowed{$_} } keys %{$mod_paths{$m}};
      next unless @paths;
      if (grep { !$path_defs{$_} } @paths) {
        push @matched_whole, "$m.$d";
        $witness_paths{$_} = 1 for grep { !$path_defs{$_} } @paths;
      } elsif (grep { $path_defs{$_}{$d} } @paths) {
        push @matched_defs, "$m.$d";
        $witness_paths{$_} = 1 for grep { $path_defs{$_}{$d} } @paths;
      }
    }
    next unless @matched_defs || @matched_whole;
    my @parts;
    push @parts, "changed definition(s) " . join(", ", @matched_defs) if @matched_defs;
    push @parts, "whole-module change covering " . join(", ", @matched_whole) if @matched_whole;
    push @hits, {
      name     => $dname,
      backends => \@eff_bks,
      has_gpu  => (grep { /^(?:cuda|hip|metal)$/ } @eff_bks) ? 1 : 0,
      paths    => [sort keys %witness_paths],
      why      => join("; ", @parts),
    };
  }
  return $declared_touch_cache{$o} = \@hits;
}
sub alias_rule_backends {   # alias -> hashref of backends explicitly pinned on its own rules, or undef
  my ($a) = @_;
  return undef unless @rule_backend_stanzas;
  my %own = map { $_ => 1 } @{$a->{own}};
  my (%bks, $any);
  for my $ds (grep { $_->{dir} eq $a->{dir} } @rule_backend_stanzas) {
    my @aliases = map { @{dfield($ds->{x}, $_) // []} } qw(alias aliases);
    my @targets = map { @{dfield($ds->{x}, $_) // []} } qw(target targets);
    next unless (grep { !ref && $_ eq $a->{alias} } @aliases)
      || (grep { !ref && $own{rpath($ds->{dir}, $_) // ""} } @targets);
    $any = 1;
    for my $rb (dfields($ds->{x}, "affected_rule_backends")) { $bks{$_} = 1 for @$rb }
  }
  return $any ? \%bks : undef;
}
sub alias_default_touches {   # alias, reached owners -> matched default records consistent with its own rules
  my ($a, @by_owner) = @_;
  return () unless @by_owner && @default_readers;
  my $rule_bks = alias_rule_backends($a);
  my (%seen, @out);
  for my $o (@by_owner) {
    for my $hit (@{declared_touch($o)}) {
      next if $seen{$hit->{name}}++;
      my @eff_bks = $rule_bks ? (grep { $rule_bks->{$_} } @{$hit->{backends}}) : @{$hit->{backends}};
      next unless @eff_bks;
      push @out, {
        %$hit,
        backends => \@eff_bks,
        has_gpu  => (grep { /^(?:cuda|hip|metal)$/ } @eff_bks) ? 1 : 0,
      };
    }
  }
  return @out;
}

# What an alias's own action reads, dune does not dump. A golden alias's
# action is a diff of two of its own targets (`NAME.expected` and the output
# beside it), whose rules are read above; any other alias's action is opaque
# -- a `(test)` stanza with no golden runs its executable under the stanza's
# `(deps)`, a hand-written rule runs whatever it names under its own `(deps)`
# -- and is printed for a changed path other than an owned module source in
# its directory, and for a changed path anywhere that is no build input (a
# script, a config, a golden: what such `(deps)` name, like the sweep
# harness's `tools/sweep.sh`). One with no owner and no targets of its own
# is unattributed and printed whenever its directory is touched at all, a
# reached stanza included, since not even its executable is known.
my %touched = map { $st[$_]{dir} => 1 } keys %reached;
$touched{(dir_of($_))[0]} = 1 for @changed;
my (%out, %unattributed, %opaque_hit, %total, %listed, %linked, %aliased, %blind_why);
my %hidden = map { $_->{name} => 1 } grep { $_->{hidden} } @al;
my %read_any = map { %{$read_by{$_}} } keys %read_by;
# The tier of each listed alias: "main"; "link", reached only by linking a
# library a changed module belongs to, never naming that module (see
# link_only); "guess", listed only because its action is unknown.
my %tier;
for my $a (@al) {
  my %owners = (%{$a->{owners}}, map { %{$runs{$_} // {}} } @{$a->{own}});
  $a->{all_owners} = \%owners;
  $aliased{$_} = 1 for keys %owners;
  $total{$a->{dir}}++ unless $a->{hidden};
  my $unknown = !%owners && !@{$a->{own}};
  my $opaque = !$a->{described};
  my $unknown_touched = $unknown && $touched{$a->{dir}};
  $unattributed{$a->{name}} = 1 if $unknown_touched && !$a->{hidden};
  my $opaque_listed = $opaque && ($any_data || $data_dirs{$a->{dir}});
  if ($opaque_listed && !$a->{hidden}) { $opaque_hit{$a->{dir}}++; $blind_why{$a->{blind_why} // "dune does not dump its action"}++ }
  my @by_owner = grep { $reached{$_} } nsort(keys %owners);
  # Uncertainty ranks first: an alias whose action is unknown and might read a
  # changed path is never link-only, whatever its owners name.
  my $blind = $unknown_touched || $opaque_listed;
  my @default_touches = alias_default_touches($a, @by_owner);
  @$a{qw(unknown_touched opaque_listed by_owner default_touches)} = ($unknown_touched, $opaque_listed, \@by_owner, \@default_touches);
  if ($dune_dirs{$a->{dir}} || $read_any{$a->{name}} || $direct{$a->{name}}) { $tier{$a->{name}} = "main" }
  elsif ($blind) { $tier{$a->{name}} = @by_owner ? "main" : "guess" }
  elsif (@by_owner) { $tier{$a->{name}} = (@default_touches || !link_only(@by_owner)) ? "main" : "link" }
  $a->{own_tier} = $tier{$a->{name}};
}
# An alias whose own rules need a listed alias (an ambient gate such as
# `runtest-env_spelling_gate`) is listed too, in that alias's tier, to a
# fixpoint: if the needed alias breaks, building the needing one fails. That
# holds for one listed only blind as well (gh-ocannl-1431 review), so a blind
# alias carries its dependents in its own tier.
my %rank = (main => 3, guess => 2, link => 1);
my %tier_of_rank = reverse %rank;
my %by_alias;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $a (@al) {
    my $now = $rank{$tier{$a->{name}} // ""} // 0;
    next if $now == 3;
    my $best = 0;
    for my $t (@{$a->{own}}) {
      for my $n (keys %{$adeps{$t} // {}}) { my $r = $rank{$tier{$n} // ""} // 0; $best = $r if $r > $best }
    }
    next unless $best > $now;
    $tier{$a->{name}} = $tier_of_rank{$best};
    $by_alias{$a->{name}} = 1;
    $grew = 1;
  }
}
for my $n (grep { !$hidden{$_} } keys %tier) {
  my ($d) = $n =~ m{^\@(?:(.*)/)?[^/]+$};
  if ($tier{$n} eq "link") { $linked{$d // "."}++; next }
  $out{$n} = 1;
  $listed{$d // "."}++;
}
my @link_only = sort grep { $tier{$_} eq "link" && !$hidden{$_} } keys %tier;

# `--why ALIAS|all` (gh-ocannl-1711): why each named alias is listed, or not,
# read from the same facts the tiers above were computed from.
sub own_target { my $b = (dir_of($_[0]))[1]; $b =~ /^__affected_probe_/ ? "its own action (read through the probe's twin $_[0])" : "the rule of its own target $_[0]" }
sub explain {
  my ($a) = @_;
  my $n = $a->{name};
  my $t = $tier{$n};
  my @r;
  push @r, $a->{hidden} ? "a helper alias that per-test aliases need, never printed itself" . ($t ? " (reached: what needs it carries its tier)" : "")
    : !$t ? "NOT LISTED: no changed path reaches it"
    : $t eq "link" ? "LINK-ONLY: printed last" . ($referenced_only ? ", and left off here by --referenced-only" : "; --referenced-only leaves it off")
    : $t eq "guess" ? "BLIND: printed first, since its action is not known here and may read a changed path"
    : "REFERENCED: printed first";
  push @r, "a dune file over its directory changed: $dune_why{$a->{dir}}" if $dune_dirs{$a->{dir}};
  push @r, "its own rules changed: $direct_why{$n}" if $direct{$n};
  for my $p (sort keys %read_via) { push @r, own_target($read_via{$p}{$n}) . " reads $p" if $read_via{$p}{$n} }
  push @r, "unattributed: no stanza of its directory has its name and it has no targets of its own, so it is listed whenever its directory is touched"
    if $a->{unknown_touched};
  if ($a->{opaque_listed}) {
    push @r, "blind (" . ($a->{blind_why} // "dune does not dump its action") . "), so listed for "
      . ($data_dirs{$a->{dir}} ? "$data_why{$a->{dir}}, a changed path in its directory or copied there"
         : "$any_data_why, a changed path no compiler or dune reads, which any action's (deps) may name");
  }
  my %owners = %{$a->{all_owners} // {}};
  for my $o (@{$a->{by_owner} // []}) {
    my @c = reach_chain($o, sub { $hit{$_[0]} });
    push @r, "its owner " . (@c > 1 ? chain_text(@c) . ", which " : label($o) . ", which ") . ($hit_why{$c[-1] // $o} // "a changed path") . " reached";
    my ($names, $why) = owner_names($o);
    push @r, ($names ? "  referenced: " : "  link-only on its account: ") . $why;
  }
  for my $dt (@{$a->{default_touches} // []}) {
    push @r, "backend-default marker `$dt->{name}` (" . join(",", @{$dt->{backends}}) . ") matches $dt->{why}";
  }
  my @idle = grep { !$reached{$_} } nsort(keys %owners);
  push @r, "no owner of it is reached: " . join(", ", map { label($_) } @idle) if @idle && !@{$a->{by_owner} // []};
  for my $t2 (@{$a->{own}}) {
    for my $m (sort grep { $tier{$_} } keys %{$adeps{$t2} // {}}) {
      push @r, own_target($t2) . " needs the alias $m, which is listed $tier{$m}" . ($by_alias{$n} ? " (it carries that tier here)" : "");
    }
  }
  return @r;
}
my @why_lines;
{
  my %by_name = map { $_->{name} => $_ } @al;
  my (@want, %seen);
  for my $w (grep { $_ ne "" } split /\n/, slurp("$tmp/why")) {
    push @want, $w eq "all" ? (sort keys %out, @link_only) : ($w =~ /^\@/ ? $w : "\@$w");
  }
  for my $w (grep { !$seen{$_}++ } @want) {
    my @r = $by_name{$w} ? explain($by_name{$w}) : ("no per-test alias dune lists has this name (`dune show aliases DIR` lists them)");
    push @why_lines, "affected: why $w: $_" for @r;
  }
}

# GPU selection is independent of reach/ranking: inspect only the alias's
# owners, never every library they link (Verdict itself declares GPU names).
# This is a file-level declaration heuristic, not OCaml scope/dataflow analysis.
my (%gpu_reason, %gpu_source_cache);
sub gpu_source {
  my ($f) = @_;
  return $gpu_source_cache{$f} if exists $gpu_source_cache{$f};
  return $gpu_source_cache{$f} = "" unless -f $f;
  my $s = code_of(slurp($f));
  my $ids = code_of($s, 1);  # comments and literal contents cannot declare on_gpu
  return $gpu_source_cache{$f} =
    $ids =~ /\bbinds_hardware_grid\b/ ? "$f names binds_hardware_grid"
    : $ids =~ /\bon_gpu\b/ ? "$f names on_gpu"
    : $ids =~ /\b(?:leg|in_leg|skipped)\b/ && ($ids =~ /\bon_(?:cuda|hip|metal)\b/ || $s =~ /"(?:cuda|hip|metal)"/)
      ? "$f declares a gated claim beside a GPU backend" : "";
}
if ($gpu_gated) {
  for my $a (grep { !$hidden{$_->{name}} && $tier{$_->{name}} } @al) {
    my $n = $a->{name};
    for my $dt (grep { $_->{has_gpu} } @{$a->{default_touches} // []}) {
      $gpu_reason{$n} //= "backend-default marker `$dt->{name}` (" . join(",", @{$dt->{backends}}) . ") matches $dt->{why}";
    }
    for my $o (sort keys %{$a->{all_owners}}) {
      for my $f (sort @{stanza_sources($o) // []}) {
        my $why = gpu_source($f);
        $gpu_reason{$n} //= $why if $why;
      }
      my %names = map { $_ => 1 } @{$st[$o]{names}};
      for my $ds (@gpu_stanzas) {
        next unless $ds->{dir} eq $st[$o]{dir} && $ds->{kind} =~ /^(?:library|executables?|tests?)$/;
        $gpu_reason{$n} //= "a GPU ocannl-backend marker on its owner" if grep { $names{$_} } dnames($ds);
      }
    }
    for my $ds (grep { $_->{dir} eq $a->{dir} && $_->{kind} eq "rule" } @gpu_stanzas) {
      my @aliases = map { @{dfield($ds->{x}, $_) // []} } qw(alias aliases);
      my @targets = map { @{dfield($ds->{x}, $_) // []} } qw(target targets);
      my %own = map { $_ => 1 } @{$a->{own}};
      $gpu_reason{$n} //= "a GPU ocannl-backend marker on its rule"
        if (grep { !ref && $_ eq $a->{alias} } @aliases)
          || (grep { !ref && $own{rpath($ds->{dir}, $_) // ""} } @targets);
    }
  }
  # Keep the unfiltered verdict below: narrowing a cross-cutting list does not
  # turn this query into evidence that prepush can safely run a smaller suite.
  print STDERR "affected: --gpu-gated: file-level declarations only; no gate dataflow or runtime coverage; unknown sources can be missed\n";
}

# An UNMAPPED module of a library this build does not compile (an optional
# backend: cuda_backend.ml on a Mac; gh-ocannl-1651): what a build that does
# compile it would reach, named on stderr for a box that builds it. Read from
# the dune files, since dune describes none of it here: the stanzas whose
# `libraries` name the library (plainly, or as a `(select)` condition, through
# other off-build libraries naming it in turn), then what requires those, to a
# fixpoint. Of the per-test aliases owned there, those whose own code names
# the module (or is not all known) are named; the rest are counted as
# link-only. Never on stdout: here they run without it.
my @off_build;   # [path, library label, [seed labels], [named aliases], number of link-only ones]
if (%unmapped_mod) {
  my %st_of;   # dune stanza -> @st indices read from it
  for my $i (0 .. $#st) { my $ds = static_of($i); push @{$st_of{$ds}}, $i if $ds }
  my $atoms; $atoms = sub { map { ref $_ ? $atoms->(@$_) : $_ } @_ };
  for my $p (sort keys %unmapped_mod) {
    my ($d, $mod) = @{$unmapped_mod{$p}};
    my @libs = grep {
      my $ds = static_of($_);
      my $m = $ds && dfield($ds->{x}, "modules");
      $ds && $st[$_]{kind} eq "library" && $st[$_]{dir} eq $d && (!$m || grep { ref $_ || $_ eq lcfirst $mod || /^:/ } @$m)
    } grep { $st[$_]{undescribed} } 0 .. $#st;
    next unless @libs;
    my %names = map { $_ => 1 } map { dnames(static_of($_)) } @libs;
    my (%seed_ds, %seeds);
    for (my $grew = 1; $grew;) {
      $grew = 0;
      for my $ds (grep { $_->{kind} =~ /^(?:library|executables?|tests?)$/ && !$seed_ds{$_} } @dstanzas) {
        next unless grep { $names{$_} } $atoms->(@{dfield($ds->{x}, "libraries") // []});
        $seed_ds{$ds} = $grew = 1;
        # Another stanza this build leaves out passes it on by its own names.
        if (grep { $st[$_]{undescribed} } @{$st_of{$ds} // []}) { $names{$_} = 1 for dnames($ds) }
        $seeds{$_} = 1 for grep { !$st[$_]{undescribed} } @{$st_of{$ds} // []};
      }
    }
    my %would = %seeds;
    for (my $grew = 1; $grew;) {
      $grew = 0;
      for my $i (0 .. $#st) {
        next if $would{$i} || !grep { $would{$_} } @{$st[$i]{to}};
        $would{$i} = $grew = 1;
      }
    }
    my (@named, $n_link);
    for my $a (grep { !$_->{hidden} } @al) {
      my @o = grep { $would{$_} } keys %{$a->{all_owners}} or next;
      if (grep { my $t = own_tokens($_); !$t || $t->{$mod} } @o) { push @named, $a->{name} } else { $n_link++ }
    }
    push @off_build, [$p, label($libs[0]), [map { label($_) } sort { $a <=> $b } keys %seeds], [sort @named], $n_link // 0];
  }
}

# A first tier spanning most of the suite (gh-ocannl-1765): the ranking cannot
# narrow this change, so say so, and who runs the whole list -- except its
# `slow-` aliases, which no CI job runs (local review): those stay local.
# A printed list spanning most of the suite over a narrower first tier
# (gh-ocannl-1996) is no pre-push run either: `low_level.ml` lists 815 of the
# 825 per-test aliases, 436 of them link-only, and that list cannot run on a
# shared box. The ranking did narrow it, so that line names the first tier as
# where the local run starts. Two thresholds, each its own function, equal today.
# The verdict (gh-ocannl-1921; the header states the contract) is decided here
# and nowhere else: the CROSS-CUTTING line, the INCOMPLETE note, the exit
# status and the trailer all read it.
sub first_tier_wide {   # (first-tier aliases, per-test aliases)
  my ($n_first, $n_all) = @_;
  return $n_all && 2 * $n_first > $n_all;
}
sub list_wide {   # (printed aliases, per-test aliases)
  my ($n_listed, $n_all) = @_;
  return $n_all && 2 * $n_listed > $n_all;
}
sub verdict {   # (first-tier aliases, printed aliases, per-test aliases, UNMAPPED paths) -> the trailer's conditions
  my ($n_first, $n_listed, $n_all, $n_unmapped) = @_;
  my @v;
  push @v, "cross-cutting" if first_tier_wide($n_first, $n_all) || list_wide($n_listed, $n_all);
  push @v, "incomplete" if $n_unmapped;
  return @v;
}
# Count observed reach from the same seeds that the ranking and --why use.
# Prefer the owner's source witnesses when a reference earned the first tier.
# A promoted alias follows a dependency in its new tier. Count a path once per alias.
sub cross_reason {
  my (%by_alias_path, %owner_path, %dir_path, %needs_alias, %count);
  for my $p (sort keys %read_by) {
    $by_alias_path{$_}{$delta_paths{$p} // $p} = 1 for keys %{$read_by{$p}};
  }
  for my $i (nsort(keys %reached)) {
    my @chain = reach_chain($i, sub { $hit_path{$_[0]} });
    next unless @chain;
    $owner_path{$i} = $hit_path{$chain[-1]};
    $dir_path{$st[$i]{dir}} //= $owner_path{$i};
  }
  $dir_path{(dir_of($_))[0]} = $_ for reverse @changed;
  for my $a (grep { $tier{$_->{name}} } @al) {
    my ($n, $d) = @$a{qw(name dir)};
    my $first = $tier{$n} ne "link";
    $needs_alias{$n} = [ sort grep { $tier{$_} && ($first ? $tier{$_} ne "link" : $tier{$_} eq "link") }
      map { keys %{$adeps{$_} // {}} } @{$a->{own}} ];
    next if !$a->{own_tier} || ($first && $a->{own_tier} eq "link");
    my @owners;
    for my $o (@{$a->{by_owner}}) {
      my ($named, undef, $witnesses) = owner_names($o);
      next if $first && !$named;
      push @owners, $first && $witnesses && @$witnesses ? @$witnesses : $owner_path{$o};
    }
    push @owners, map { @{$_->{paths}} } @{$a->{default_touches} // []} if $first;
    my $paths = $by_alias_path{$n} //= {};
    my $data_path = $data_why{$d} // $any_data_why;
    $data_path = $delta_paths{$data_path} // $data_path if defined $data_path;
    $paths->{$_} = 1 for grep { defined }
      $dune_why{$d}, $direct_path{$n}, @owners,
      ($a->{opaque_listed} ? $data_path : ()),
      ($a->{unknown_touched} ? $dir_path{$d} : ());
  }
  for my $n (@_) {
    my @queue = ($n);
    my %seen;
    while (defined(my $alias = shift @queue)) {
      next if $seen{$alias}++;
      my @paths = keys %{$by_alias_path{$alias} // {}};
      if (@paths) { $count{$_}++ for @paths; last }
      push @queue, @{$needs_alias{$alias} // []};
    }
  }
  my @top = sort { $count{$b} <=> $count{$a} || $a cmp $b } keys %count;
  splice @top, 3 if @top > 3;
  return join(", ", map { sprintf "%s (%d aliases)", s/[^[:print:]]/?/gr, $count{$_} } @top);
}
my $n_all = 0;
$n_all += $_ for values %total;
my $n_first = keys %out;
my $n_listed = $n_first + ($referenced_only ? 0 : @link_only);
my %verdict = map { $_ => 1 } verdict($n_first, $n_listed, $n_all, scalar @unmapped);
if ($verdict{"cross-cutting"}) {
  my $first_wide = first_tier_wide($n_first, $n_all);
  my $reason = cross_reason(keys %out, $first_wide ? () : @link_only);
  # FD 3 is the original stderr, before the shell buffered other diagnostics.
  open my $first, ">&=3" or die "cannot open first diagnostic: $!";
  printf $first "affected: CROSS-CUTTING: %s spans most of the suite (%d of %d aliases)\n",
    $first_wide ? "first tier" : "full list", $first_wide ? $n_first : $n_listed, $n_all;
  printf $first "affected: CROSS-CUTTING reason: %s\n", $reason if length $reason;
  close $first or die "cannot write first diagnostic: $!";
}

my @selected = grep { !$gpu_gated || $gpu_reason{$_} }
  (sort(keys %out), $referenced_only ? () : @link_only);
print "$_\n" for @selected;
if ($gpu_gated) {
  print STDERR "affected: gpu-gated: $_: $gpu_reason{$_}\n" for @selected;
  printf STDERR "affected: --gpu-gated: %d selected alias(es); counts and verdict below describe unfiltered reach\n", scalar @selected;
}
my $n_by_alias = grep { $out{$_} } keys %by_alias;
printf STDERR "affected: %d alias(es) listed because their rules need a listed alias\n", $n_by_alias if $n_by_alias;
printf STDERR "affected: %s: %d of its %d per-test aliases%s\n", $_, $listed{$_} // 0, $total{$_},
  ($linked{$_} ? " (and $linked{$_} link-only)" : "") for sort keys %{{ %listed, %linked }};
my %bare;
$bare{$st[$_]{dir}}++ for grep { !$aliased{$_} && $st[$_]{kind} ne "library" } keys %reached;
printf STDERR "affected: %s: %d reached executable stanza(s) no per-test alias runs\n", $_, $bare{$_} for sort keys %bare;
# Only worth saying when a changed module has no stanza here: it may belong to one of these.
if (@unmapped) {
  print STDERR "affected: optional library $_ is not in this build; edges to it skipped\n" for sort keys %skipped_optional;
  print STDERR "affected: library $_ is not in this build, nor is the stanza requiring it; edge skipped\n" for sort keys %skipped_off_build;
}
print STDERR "affected: listed as unattributed: $_ (no stanza of its directory has that name, no targets of its own; its directory is touched)\n" for sort keys %unattributed;
printf STDERR "affected: %s: %d blind per-test alias(es) (actions not described here), listed for the changed paths there\n", $_, $opaque_hit{$_} for sort keys %opaque_hit;
printf STDERR "affected: %d of them blind because %s\n", $blind_why{$_}, $_ for sort keys %blind_why;
print STDERR "affected: unread: $_ (no stanza owns it and no rule dune shows reads it; only blind aliases are listed for it)\n" for @unread;
print STDERR "affected: UNMAPPED: $_ (no stanza of this build owns this module)\n" for @unmapped;
for my $ob (@off_build) {
  my ($p, $lib, $seeds, $named, $n_link) = @$ob;
  my $n = @$named + $n_link;
  printf STDERR "affected: OFF-BUILD: %s is a module of %s, which this build does not compile. Where it is compiled, %s"
    . " link%s it (by name or a (select)), reaching %d per-test alias(es): run them on a box that builds it, with its backend"
    . " selected, or run `affected` there for the ranked list. %d of them name the module in their own code (or hold code"
    . " not all known)%s, %d only link it%s\n",
    $p, $lib, (@$seeds ? (@$seeds > 3 ? scalar(@$seeds) . " stanzas" : join(", ", @$seeds)) : "no stanza"), (@$seeds == 1 ? "s" : ""),
    $n, scalar(@$named), (@$named ? ", named below" : ""), $n_link, (@$seeds ? "" : "; nothing here links it");
  print STDERR "affected: off-build: $_\n" for @$named;
}
if (@link_only) {
  my $what = join("; ", map { $_ . (!$mod_defs{$_} ? "" : %{$mod_defs{$_}} ? " (" . join(", ", sort keys %{$mod_defs{$_}}) . ")"
                                    : " (no definition: only comments or blank lines changed)") }
                        sort keys %mod_defs);
  printf STDERR "affected: %d LINK-ONLY alias(es), %s: each links the library of a changed module, %s, but no code of"
    . " its own names the changed part, so it is reached only through other library code. A change in what that"
    . " code does for its callers still breaks them (gh-ocannl-1235: Register_tile); run them before pushing such a"
    . " change. %s\n",
    scalar(@link_only), $referenced_only ? "NOT printed (--referenced-only)" : "printed LAST, after the others",
    $what, $referenced_only ? "They are:" : "--referenced-only leaves them off. They are:";
  print STDERR "affected: link-only: $_\n" for @link_only;
}
print STDERR "$_\n" for @why_lines;
# The harness legs (gh-ocannl-1900): what each changed path reached, and the
# commands; `--harnesses` prints those on stdout instead. The closing summary
# counts the commands (gh-ocannl-1929).
my $n_harness = 0;
{
  for my $f ("harness-err", "harness-cmds") {
    open my $fh, "<", "$tmp/$f" or next;
    while (my $l = <$fh>) {
      print STDERR $f eq "harness-cmds" ? "affected: harness: $l" : $l;
      $n_harness++ if $f eq "harness-cmds";
    }
    close $fh;
  }
}
{
  my @slow = sort grep { m{(?:^\@|/)slow-[^/]*$} } keys %out, @link_only;   # link-only ones too: CI runs none
  my $slow = @slow ? sprintf(" but its %d slow- alias(es), which no CI job runs: run those locally too", scalar @slow) : "";
  if (!$verdict{"cross-cutting"}) {
  } elsif (first_tier_wide($n_first, $n_all)) {
    printf STDERR "affected: CROSS-CUTTING: %d of the %d per-test aliases rank first, so this change is cross-cutting:"
      . " a full-suite run on a CPU-only box before the merge owns the full list%s; locally run the tests that exercise the change, named in"
      . " the PR body with that run, and `dune build \@check`\n", $n_first, $n_all, $slow;
  } else {
    printf STDERR "affected: CROSS-CUTTING: %d of the %d per-test aliases are listed, %d of them link-only, so this change"
      . " is cross-cutting by its list, though only %d rank first: a full-suite run on a CPU-only box before the merge owns the full list%s;"
      . " locally run the tests that exercise the change, named in the PR body with that run -- the %d that rank first"
      . " (--referenced-only) are where to start -- and `dune build \@check`\n", $n_listed, $n_all, scalar(@link_only), $n_first, $slow, $n_first;
  }
}
printf STDERR "affected: %d alias(es) from %d changed path(s)%s%s%s\n", scalar(keys %out) + ($referenced_only ? 0 : @link_only),
  scalar(@changed), (@link_only ? sprintf(", %d of them link-only%s", scalar(@link_only), $referenced_only ? " and left off" : "") : ""),
  ($n_harness ? sprintf(", and %d harness command(s), named above", $n_harness) : ""),
  $verdict{incomplete} ? "; INCOMPLETE, see UNMAPPED above" : "";
# The trailer: the last line of stderr, one line, in this order.
printf STDERR "affected: verdict=%s\n", join(",", grep { $verdict{$_} } qw(cross-cutting incomplete)) || "ok";
exit($verdict{incomplete} ? 3 : 0);
PERL
rc=0
perl "$tmp/main.pl" "$tmp" "$referenced_only" "$probe_ok" "$qroot" "$tmp/probe-old" "$gpu_gated" || rc=$?
# 4: the probe loaded, but dune could not print a rule there (an input the copy
# lacks, say): the whole reading again, without the probe.
if [ "$rc" = 4 ] && [ "$probe_ok" = 1 ]; then
  probe_ok=0
  qroot=.
  for x in manifest forms manifest-old forms-old; do : >"$tmp/$x"; done
  if [ -s "$tmp/scoped" ]; then
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "the probe failed" >>"$tmp/unscoped"; done <"$tmp/scoped"
    : >"$tmp/scoped"
  fi
  listings
  rc=0
  perl "$tmp/main.pl" "$tmp" "$referenced_only" "$probe_ok" "$qroot" "$tmp/probe-old" "$gpu_gated" || rc=$?
fi
exit "$rc"
