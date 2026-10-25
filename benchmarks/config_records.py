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
"""

import strict_json

TAG = '{"ocannl_config":'
VERSION = 1
SOURCES = ("commandline", "environment", "config_file", "profile", "default")
LAYERS = ("commandline", "environment", "config_file", "default")
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
        consistent = layer != "default" and bool(profile) and origin is not None
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
