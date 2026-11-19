"""The reader of the library's resolved-configuration stream (gh-ocannl-1566), stdlib only.

An OCANNL process run with `--ocannl_log_config_json=true` prints, on stderr, one JSON object per
setting it resolves -- the records `Utils.resolved_config ()` returns, rendered by
`Utils.config_record_to_json` (schema in `ocannl_config.reference`, key `log_config_json`):

    {"ocannl_config":1,"key":K,"value":V,"source":S,"layer":L,"profile":P,"origin":O}

`parse_config_records(text)` picks those lines out of a process's stderr and returns
`{key: {"value", "source", "layer", "profile", "origin"}}`. Every other line is the stream's other
traffic and is ignored; a line that opens with the tag but does not hold a valid record -- a write
another writer tore, an emitter that regressed -- is refused with `ConfigRecordError` rather than
skipped, as is a schema version this reader does not know, and one key resolved two different
ways (the process's sources changed under it: whatever it measured ran under both). The stream
is point-in-time: a process killed early leaves a truthful but incomplete one.

`expect_problems(config, expects)` is the check a driver states a treatment with: each named key
resolved to its value from the command line. Run as a script, `config_records.py expect FILE
KEY=VALUE...` applies it to a stderr file a shell driver kept, and refuses as well any setting that
entered through the environment: exit 0 when the treatment holds, 1 naming each problem, 2 for a
usage error. `config_records.py show FILE KEY...` prints each key's value and provenance for the
driver's log (`(not resolved)` for one the process never read). They are how a driver that keeps
its own cells confirms and records what they ran under, in place of reading the prose
configuration trace (gh-ocannl-1607).
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    # A shell driver runs this under `python3 -I`, which leaves the script's own directory off
    # sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_json  # noqa: E402

TAG = '{"ocannl_config":'
VERSION = 1
# `in_process` is a program's own statement (Utils.state_config, gh-ocannl-2354), above the rest.
SOURCES = ("in_process", "commandline", "environment", "config_file", "profile", "default")
LAYERS = ("in_process", "commandline", "environment", "config_file", "default")
FIELDS = ("ocannl_config", "key", "value", "source", "layer", "profile", "origin")


class ConfigRecordError(ValueError):
    """A tagged line of the stream is not a record this reader can vouch for."""


def _record(line):
    try:
        obj = strict_json.loads(line)
    except ValueError as e:
        raise ConfigRecordError(f"a tagged line that is not a JSON record ({e}): {line!r}") from None
    if not isinstance(obj, dict) or tuple(obj) != FIELDS:
        raise ConfigRecordError(f"a record without exactly the fields {FIELDS}: {line!r}")
    if obj["ocannl_config"] != VERSION or isinstance(obj["ocannl_config"], bool):
        raise ConfigRecordError(f"schema version {obj['ocannl_config']!r}, not {VERSION}: {line!r}")
    for name in ("key", "value", "source", "layer"):
        if not isinstance(obj[name], str):
            raise ConfigRecordError(f"field {name} is not a string: {line!r}")
    for name in ("profile", "origin"):
        if obj[name] is not None and not isinstance(obj[name], str):
            raise ConfigRecordError(f"field {name} is neither a string nor null: {line!r}")
    source, layer, profile, origin = obj["source"], obj["layer"], obj["profile"], obj["origin"]
    if source not in SOURCES or layer not in LAYERS:
        raise ConfigRecordError(f"an unknown source or layer: {line!r}")
    # The combinations Utils.config_record_to_json can produce, and no other.
    if source == "profile":
        # No profile is picked in process: Utils.state_config refuses the `profile` key.
        consistent = layer not in ("default", "in_process") and bool(profile) and origin is not None
    elif source == "in_process":
        consistent = layer == source and profile is None and origin == "Utils.state_config"
    elif source == "default":
        consistent = layer == "default" and profile is None and origin is None
    else:
        consistent = layer == source and profile is None and bool(origin)
    if not consistent:
        raise ConfigRecordError(f"source {source!r} does not fit its layer, profile and origin: "
                                f"{line!r}")
    if not obj["key"]:
        raise ConfigRecordError(f"a record with an empty key: {line!r}")
    return obj["key"], {"value": obj["value"], "source": source, "layer": layer,
                        "profile": profile, "origin": origin}


def parse_config_records(text):
    """{key: {"value", "source", "layer", "profile", "origin"}} from the `log_config_json` records
    in `text` (a process's stderr), refusing a malformed tagged line, an unknown version, or one key
    with two different records with `ConfigRecordError`."""
    config = {}
    # Records are framed by "\n" alone: str.splitlines would also split at U+0085, U+2028 and
    # U+2029, which the library's encoder leaves inside a string as the UTF-8 they are.
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line.startswith(TAG):
            continue
        key, entry = _record(line)
        known = config.setdefault(key, entry)
        if known != entry:
            raise ConfigRecordError(f"ocannl_{key} resolved two ways: {known} and then {entry}")
    return config


def provenance(entry):
    """A short human reading of a record's provenance, e.g. `commandline --ocannl_backend=cc`,
    `profile approximate via commandline --ocannl_profile=approximate`, `default`."""
    if entry["source"] == "default":
        return "default"
    if entry["source"] == "profile":
        return f"profile {entry['profile']} via {entry['layer']} {entry['origin']}"
    return f"{entry['source']} {entry['origin']}"


def expect_problems(config, expects):
    """Why `config` (`parse_config_records`) does not state each of `expects` ({key: value}): the key
    was not read, resolved to another value, or came from somewhere other than the command line."""
    problems = []
    for key, value in expects.items():
        entry = config.get(key)
        if entry is None:
            problems.append(f"ocannl_{key} was never resolved: the treatment {value!r} did not take")
        elif entry["value"] != value or entry["source"] != "commandline":
            problems.append(f"ocannl_{key} resolved to {entry['value']!r} from {provenance(entry)}, "
                            f"not to {value!r} from the command line")
    return problems


def parse_expects(words):
    """KEY=VALUE words as {key: value}; ValueError for a word that is not one, or a key twice."""
    expects = {}
    for word in words:
        key, sep, value = word.partition("=")
        if not sep or not key or key.startswith("-"):
            raise ValueError(f"{word!r}: expected KEY=VALUE, KEY a setting without its ocannl_ prefix")
        if key in expects:
            raise ValueError(f"{key} is expected twice")
        expects[key] = value
    return expects


def main(argv):
    if len(argv) < 3 or argv[0] not in ("expect", "show"):
        print("usage: config_records.py expect FILE KEY=VALUE... | show FILE KEY...",
              file=sys.stderr)
        return 2
    try:
        expects = parse_expects(argv[2:]) if argv[0] == "expect" else dict.fromkeys(argv[2:])
    except ValueError as e:
        print(f"config_records: {e}", file=sys.stderr)
        return 2
    try:
        with open(argv[1], encoding="utf-8", errors="replace") as f:
            config = parse_config_records(f.read())
    except OSError as e:
        print(f"config_records: {argv[1]}: {e}")
        return 1
    except ConfigRecordError as e:
        print(f"config_records: {argv[1]}: unreadable configuration records: {e}")
        return 1
    if argv[0] == "show":
        for key in expects:
            entry = config.get(key)
            print(f"ocannl_{key}: " + ("(not resolved)" if entry is None else
                                        f"{entry['value']!r} ({provenance(entry)})"))
        return 0
    problems = expect_problems(config, expects)
    problems += [f"ocannl_{key} came from the environment" for key, entry in sorted(config.items())
                 if entry["layer"] == "environment"]
    for p in problems:
        print(f"config_records: {argv[1]}: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
