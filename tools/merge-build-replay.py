#!/usr/bin/env python3
"""Replay a branch's merge history against merge-build input sets, to size the boundary.

Usage: tools/merge-build-replay.py [--since DATE] [--until DATE] [--set NAME[=GLOB,...]]...
                                   [--trigger same-input|any-input] [--top N] [--repo DIR]
                                   [--verbose] [REV...]

For every first-parent two-parent merge `git log --first-parent --merges REV...` lists (REV
defaults to HEAD; `--since`/`--until` pass through to git log, and a bare date there takes the
current time of day, so spell a time to make a replay repeatable), it reads the paths each side
changed since the merge base, as ship-pr's merge build does (`.github/ship-pr-merge-build`'s
header): both names of a rename (git's -M detection), a path a rename on either side touched
counting by its other name too. Then, per input set, it decides whether the merge build would
have run on that merge:
  same-input  a path BOTH sides changed matches one of the set's globs;
  any-input   EACH side changed some path matching one (the same or not).
A glob matches the WHOLE repository path under Python's fnmatch.fnmatchcase -- case-sensitive,
`*` crossing `/` -- the merge build's own matcher. The replay assumes every merge's base had
moved since its head's CI run, so it bounds the real build count from above.

Input sets. `--set NAME=GLOB,GLOB,...` names a set (a glob cannot contain a comma; repeating a
NAME appends to it); a `@OTHER` element includes the globs of a set defined earlier or of a
built-in one. `--set NAME` alone selects a built-in set, read from where it is defined in the
replayed repository's working tree, never copied here:
  config      the `input` lines of `.github/ship-pr-merge-build` (what triggers the merge build);
  shell-tier  `SHELL_GLOB` in `tools/merge-build-check.sh` (what selects its shell tier, which
              that script decides on paths both sides changed: replay it under same-input);
              as the script does, it also builds every merge where git would QUOTE a path either
              side changed (a double quote, a backslash or a control character in
              it), whatever the globs say.
With no `--set`, both built-ins are replayed. `--trigger` defaults to the config's `trigger`
line (same-input when it has none, or when there is no config).

Output, per set in the order given: the merges replayed, how many would build, and the paths that
triggered them, counted by merges (`--top N`, default 15, 0 for all); from the second set on,
only the merges it builds beyond the PREVIOUS set, so a widening sequence of sets lists what each
step adds. A Markdown summary table closes the report. `--verbose` also prints each merge with the
sets that would build it.

Exit status: 0 the replay ran; 1 a git command failed; 2 a usage error, or a built-in set whose
definition could not be read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import os
import re
import subprocess
import sys

ME = "merge-build-replay"
CONFIG_PATH = ".github/ship-pr-merge-build"
CHECK_PATH = "tools/merge-build-check.sh"
TRIGGERS = ("same-input", "any-input")


class Usage(Exception):
    pass


class GitFailed(Exception):
    pass


# Git paths are bytes, not text in any encoding: decode them as UTF-8 with surrogateescape, so a
# path that is not valid UTF-8 round-trips to stdout byte for byte instead of failing the replay.
GIT_TEXT = {"encoding": "utf-8", "errors": "surrogateescape"}


def git(repo: str, *args: str) -> str:
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, **GIT_TEXT)
    if done.returncode != 0:
        raise GitFailed(f"git {' '.join(args)}: exit {done.returncode}: {done.stderr.strip()}")
    return done.stdout


def read_text(repo: str, rel: str) -> str:
    try:
        with open(os.path.join(repo, rel), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise Usage(f"cannot read {rel} in {repo}: {e.strerror}") from e


def config_definition(repo: str) -> tuple[list[str], str | None]:
    """The config's `input` globs and its `trigger` (None when it names none), by the merge
    build's grammar: `#` comments and blank lines skipped, `KEY VALUE` lines."""
    inputs: list[str] = []
    trigger = None
    for raw in read_text(repo, CONFIG_PATH).split("\n"):
        line = raw.rstrip("\r").strip(" \t")
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        value = value.strip(" \t")
        if key == "input" and value:
            inputs.append(value)
        elif key == "trigger" and value:
            trigger = value
    if not inputs:
        raise Usage(f"{CONFIG_PATH} has no `input` line")
    return inputs, trigger


def shell_tier_definition(repo: str) -> list[str]:
    found = re.findall(r"^SHELL_GLOB='([^']+)'$", read_text(repo, CHECK_PATH), re.M)
    if len(found) != 1:
        raise Usage(f"{CHECK_PATH} has {len(found)} `SHELL_GLOB='...'` assignment(s), not exactly one")
    return found


def builtin_globs(repo: str, name: str) -> list[str]:
    if name == "config":
        return config_definition(repo)[0]
    if name == "shell-tier":
        return shell_tier_definition(repo)
    raise Usage(f"no set named `{name}` (the built-in sets are `config` and `shell-tier`)")


def resolve_sets(repo: str, specs: list[str]) -> list[tuple[str, list[str], str]]:
    """Each set's NAME, globs and where they came from, in first-mention order."""
    order: list[str] = []
    globs: dict[str, list[str]] = {}
    source: dict[str, str] = {}
    for spec in specs:
        name, eq, body = spec.partition("=")
        if not name or "," in name or "@" in name:
            raise Usage(f"--set {spec!r}: a set needs a NAME without `,` or `@`")
        if not eq:
            if name in globs:
                raise Usage(f"--set {name}: selected or defined twice")
            order.append(name)
            globs[name] = builtin_globs(repo, name)
            source[name] = CONFIG_PATH if name == "config" else CHECK_PATH
            continue
        if name not in globs:
            order.append(name)
            globs[name] = []
            source[name] = "--set"
        elif source[name] != "--set":
            raise Usage(f"--set {spec!r}: `{name}` is a built-in set already selected")
        elements = body.split(",")
        if any(not e for e in elements):
            raise Usage(f"--set {spec!r}: an empty glob")
        for e in elements:
            if e.startswith("@"):
                other = e[1:]
                if other == name:
                    raise Usage(f"--set {spec!r}: a set cannot include itself")
                globs[name].extend(globs[other] if other in globs else builtin_globs(repo, other))
            else:
                globs[name].append(e)
    return [(n, globs[n], source[n]) for n in order]


def side(repo: str, a: str, b: str) -> tuple[set[str], dict[str, list[str]]]:
    """Every path the diff names (both names of a rename) and each renamed name's other name."""
    tokens = git(repo, "diff", "--no-ext-diff", "-M", "--name-status", "-z", a, b).split("\0")
    names: set[str] = set()
    other: dict[str, list[str]] = {}
    i = 0
    while i < len(tokens) and tokens[i]:
        status = tokens[i]
        if status[0] in "RC":
            old, new = tokens[i + 1], tokens[i + 2]
            names |= {old, new}
            other.setdefault(old, []).append(new)
            other.setdefault(new, []).append(old)
            i += 3
        else:
            names.add(tokens[i + 1])
            i += 2
    return names, other


def git_quotes(path: str) -> bool:
    """Whether `git -c core.quotePath=false diff --name-status` would quote the path: a double
    quote, a backslash or a control character in it."""
    return any(c in '"\\' or ord(c) < 0x20 or ord(c) == 0x7F for c in path)


def matching(paths, globs: list[str], other: dict[str, list[str]]) -> list[str]:
    return sorted(
        p for p in paths if any(fnmatch.fnmatchcase(n, g) for n in (p, *other.get(p, [])) for g in globs)
    )


def main(argv: list[str]) -> int:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(
        prog="tools/merge-build-replay.py",
        usage=argparse.SUPPRESS,
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument("revs", nargs="*", metavar="REV")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--set", dest="sets", action="append", default=[])
    parser.add_argument("--trigger", choices=TRIGGERS)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.top < 0:
        raise Usage("--top takes a count, 0 for all")
    repo = git(args.repo, "rev-parse", "--show-toplevel").strip()

    sets = resolve_sets(repo, args.sets or ["config", "shell-tier"])
    trigger, trigger_from = args.trigger, "--trigger"
    if trigger is None:
        trigger, trigger_from = "same-input", "the default: no config"
        if os.path.exists(os.path.join(repo, CONFIG_PATH)):
            named = config_definition(repo)[1]
            trigger, trigger_from = (named, CONFIG_PATH) if named else ("same-input", f"the default: {CONFIG_PATH} names none")
        if trigger not in TRIGGERS:
            raise Usage(f"{CONFIG_PATH}'s trigger `{trigger}` is neither {' nor '.join(TRIGGERS)}")

    log_args = ["log", "--first-parent", "--merges", "--format=%H%x09%P%x09%s"]
    if args.since:
        log_args.append(f"--since={args.since}")
    if args.until:
        log_args.append(f"--until={args.until}")
    log_args += [*(args.revs or ["HEAD"]), "--"]
    listed = [line.split("\t", 2) for line in git(repo, *log_args).splitlines() if line]

    replayed = 0
    not_replayed: Counter[str] = Counter()
    builds = {name: 0 for name, _, _ in sets}
    beyond = {name: 0 for name, _, _ in sets}
    triggered = {name: Counter() for name, _, _ in sets}
    per_merge: list[str] = []
    for sha, parents, subject in reversed(listed):
        ps = parents.split()
        if len(ps) != 2:
            not_replayed[f"{len(ps)} parents"] += 1
            continue
        # Exit 1 with nothing said is "no common ancestor"; anything else (a missing object in a
        # shallow or damaged clone, exit 128) is a failed read, not a merge to leave out.
        found = subprocess.run(["git", "-C", repo, "merge-base", ps[0], ps[1]], capture_output=True, **GIT_TEXT)
        if found.returncode == 1 and not found.stdout.strip() and not found.stderr.strip():
            not_replayed["no merge base"] += 1
            continue
        if found.returncode != 0:
            raise GitFailed(f"git merge-base {ps[0]} {ps[1]}: exit {found.returncode}: {found.stderr.strip()}")
        mb = found.stdout.strip()
        replayed += 1
        base, base_other = side(repo, mb, ps[0])
        head, head_other = side(repo, mb, ps[1])
        other = {p: base_other.get(p, []) + head_other.get(p, []) for p in base | head}
        quoted = sorted(p for p in base | head if git_quotes(p))
        built_by = []
        previous = None
        for name, globs, src in sets:
            if src == CHECK_PATH and quoted:
                # merge-build-check.sh fails closed: a path git quotes on either side runs every tier.
                hits, builds_here = quoted, True
            elif trigger == "same-input":
                hits = matching(base & head, globs, other)
                builds_here = bool(hits)
            else:
                mine, theirs = matching(head, globs, other), matching(base, globs, other)
                hits = sorted(set(mine) | set(theirs))
                builds_here = bool(mine) and bool(theirs)
            if builds_here:
                builds[name] += 1
                built_by.append(name)
                if previous is None or previous not in built_by:
                    if previous is not None:
                        beyond[name] += 1
                    triggered[name].update(hits)
            previous = name
        if args.verbose:
            per_merge.append(f"{sha[:10]} {subject[:72]}: {' '.join(built_by) or '-'}")

    revs = " ".join(args.revs or ["HEAD"])
    window = "".join(f" --{k} {v!r}" for k, v in (("since", args.since), ("until", args.until)) if v)
    skipped = "; not replayed: " + ", ".join(f"{n} with {why}" for why, n in sorted(not_replayed.items())) if not_replayed else ""
    print(f"{ME}: {len(listed)} first-parent merge(s) of {revs}{window}, {replayed} replayed{skipped}")
    print(f"{ME}: trigger {trigger} ({trigger_from})")
    if per_merge:
        print("\nmerges, oldest first, with the sets that would build each:")
        for line in per_merge:
            print(f"  {line}")
    previous = None
    for name, globs, src in sets:
        print(f"\nset {name}: {' '.join(globs)}  ({src})")
        print(f"  replayed {replayed}, would build {builds[name]}")
        if previous is None:
            print("  paths that triggered them, by merges:")
        else:
            print(f"  beyond {previous}: {beyond[name]} merge(s); the paths that triggered those, by merges:")
        common = triggered[name].most_common(args.top or None)
        for path, n in common:
            print(f"    {n:5d}  {path}")
        rest = len(triggered[name]) - len(common)
        if rest > 0:
            print(f"    ... {rest} more path(s) (--top 0 lists all)")
        previous = name
    print("\n| set | globs | replayed | would build | beyond the previous set |")
    print("| --- | --- | --- | --- | --- |")
    previous = None
    for name, globs, _ in sets:
        shown = " ".join(f"`{g}`" for g in globs)
        print(f"| {name} | {shown} | {replayed} | {builds[name]} | {'-' if previous is None else beyond[name]} |")
        previous = name
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
        sys.exit(main(sys.argv[1:]))
    except Usage as e:
        print(f"{ME}: {e}", file=sys.stderr)
        sys.exit(2)
    except GitFailed as e:
        print(f"{ME}: {e}", file=sys.stderr)
        sys.exit(1)
