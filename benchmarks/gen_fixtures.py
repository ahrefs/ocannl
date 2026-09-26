#!/usr/bin/env python3
"""Generate self-describing safetensors fixtures for the cross-framework benchmark suite.

Each fixture holds the initial weights, the full dataset (inputs and one-hot labels), and
the workload hyperparameters in the safetensors __metadata__ map, so every runner needs
only the fixture path. All payloads are float32; weights are [fan_out, fan_in] row-major
(the shared convention: PyTorch nn.Linear layout, OCANNL output-axes-then-input-axes).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import fixture_digest


def gen_moons(rng, n, noise=0.1):
    """Two interleaved half-moons, n samples, inputs [n, 2], labels [n] in {0, 1}."""
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0.0, np.pi, n0)
    t1 = rng.uniform(0.0, np.pi, n1)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    x = np.concatenate([x0, x1]).astype(np.float32)
    x += rng.normal(0.0, noise, x.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n0, np.int64), np.ones(n1, np.int64)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def gen_gaussian(rng, n, din, num_classes):
    x = rng.normal(0.0, 1.0, (n, din)).astype(np.float32)
    y = rng.integers(0, num_classes, n)
    return x, y


def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes), np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def uniform(rng, fan_in, shape):
    scale = np.sqrt(1.0 / fan_in)
    return rng.uniform(-scale, scale, shape).astype(np.float32)


def build_mlp(spec, rng, tensors, meta):
    dims = spec["dims"]
    total = spec["batch_size"] * spec["n_batches"]
    for i, (din, dout) in enumerate(zip(dims, dims[1:]), start=1):
        tensors[f"w{i}"] = uniform(rng, din, (dout, din))
        tensors[f"b{i}"] = np.zeros(dout, np.float32)
    if spec["data"] == "moons":
        assert dims[0] == 2 and dims[-1] == 2, "moons data is 2-D input, 2 classes"
        x, y = gen_moons(rng, total)
    elif spec["data"] == "gaussian":
        x, y = gen_gaussian(rng, total, dims[0], dims[-1])
    else:
        raise ValueError(f"unknown data kind {spec['data']!r}")
    tensors["x"] = x
    tensors["y"] = one_hot(y, dims[-1])
    meta["n_layers"] = str(len(dims) - 1)


def build_conv(spec, rng, tensors, meta):
    """A two-conv classifier. Weight layouts follow OCANNL's axis order (output axes then
    input axes, channels-last images): conv kernel [oc, kh, kw, ic]; fc1 weight
    [hid, oh, ow, oc] (input axes = the conv feature map); images [total, h, w, c]. The
    Python runners permute to NCHW.

    Three shapes drive the same builder: LeNet-5 (1-channel, valid convs, small channels),
    the cifar-scale variant (gh-ocannl-500: 3-channel 44x44, valid 5x5 convs so both conv
    GEMM rows land on multiples of 8 — 40 and 16 — and channels multiples of 8, so the
    blocked conv sketch legs' per-block tile gates fire), and the stride-2-stem variant
    (gh-ocannl-502: 3-channel 51x51, conv1 at stride 2 — the strided downsampling site the
    compacting Stage targets — with rows 24 and 8 still multiples of 8). [in_channels]
    defaults to 1, [use_padding] to false, and [stride1]/[stride2] to 1, keeping the LeNet
    fixture bit-identical. Strides are only supported with valid convs: same-padding
    output-size conventions differ across frameworks once stride > 1."""
    total = spec["batch_size"] * spec["n_batches"]
    img, classes = spec["image_size"], spec["classes"]
    c1, c2, k = spec["channels1"], spec["channels2"], spec["kernel_size"]
    ic = spec.get("in_channels", 1)
    use_padding = spec.get("use_padding", False)
    s1, s2 = spec.get("stride1", 1), spec.get("stride2", 1)
    if use_padding:
        assert s1 == 1 and s2 == 1, "strides require use_padding: false"
        # Same-padding keeps the spatial extent; two 2x pools halve it twice.
        fm = img // 2 // 2
    else:
        # conv-valid at stride, pool2, conv-valid at stride, pool2. OCANNL's valid conv
        # (and pool) require exact divisibility — assert instead of silently flooring.
        assert (img - k) % s1 == 0, "conv1: (image_size - kernel_size) mod stride1 != 0"
        c1_out = (img - k) // s1 + 1
        assert c1_out % 2 == 0, "pool1: conv1 output extent must be even"
        p1 = c1_out // 2
        assert (p1 - k) % s2 == 0, "conv2: (input - kernel_size) mod stride2 != 0"
        c2_out = (p1 - k) // s2 + 1
        assert c2_out % 2 == 0, "pool2: conv2 output extent must be even"
        fm = c2_out // 2
    fc1, fc2 = spec["fc1"], spec["fc2"]
    tensors["conv1_kernel"] = uniform(rng, k * k * ic, (c1, k, k, ic))
    tensors["conv1_bias"] = np.zeros(c1, np.float32)
    tensors["conv2_kernel"] = uniform(rng, k * k * c1, (c2, k, k, c1))
    tensors["conv2_bias"] = np.zeros(c2, np.float32)
    tensors["fc1_w"] = uniform(rng, fm * fm * c2, (fc1, fm, fm, c2))
    tensors["fc1_b"] = np.zeros(fc1, np.float32)
    tensors["fc2_w"] = uniform(rng, fc1, (fc2, fc1))
    tensors["fc2_b"] = np.zeros(fc2, np.float32)
    tensors["w_logits"] = uniform(rng, fc2, (classes, fc2))
    tensors["b_logits"] = np.zeros(classes, np.float32)
    tensors["x"] = rng.normal(0.0, 1.0, (total, img, img, ic)).astype(np.float32)
    tensors["y"] = one_hot(rng.integers(0, classes, total), classes)
    for key in ("image_size", "classes", "channels1", "channels2", "kernel_size", "fc1", "fc2"):
        meta[key] = str(spec[key])
    meta["in_channels"] = str(ic)
    meta["use_padding"] = "true" if use_padding else "false"
    meta["stride1"] = str(s1)
    meta["stride2"] = str(s2)


def build_gpt(spec, rng, tensors, meta):
    """GPT-2-style decoder. The same builder serves the forward-only workload (gpt2_mini,
    [mode: infer]) and the training one (gpt2_mini_train, [mode: train], gh-ocannl-551),
    which trains every weight below with plain SGD. OCANNL layouts:
    wte [d_model, vocab] (output d, input v — used transposed as the tied lm_head);
    wq/wk/wv [heads, d_head, d_model]; wo [d_model, heads, d_head];
    ffn w1 [d_ff, d_model], w2 [d_model, d_ff]; ln gammas/betas [d_model];
    wpe [seq, d_model]. Token ids and CE-target ids as integral float32 [total, seq]."""
    total = spec["batch_size"] * spec["n_batches"]
    d, v, seq = spec["d_model"], spec["vocab"], spec["seq_len"]
    nh, dh, dff = spec["n_head"], spec["d_model"] // spec["n_head"], spec["d_ff"]
    tensors["wte"] = uniform(rng, d, (d, v))
    tensors["wpe"] = (0.01 * rng.normal(0.0, 1.0, (seq, d))).astype(np.float32)
    for i in range(spec["n_layer"]):
        for w in ("wq", "wk", "wv"):
            tensors[f"l{i}_{w}"] = uniform(rng, d, (nh, dh, d))
        tensors[f"l{i}_wo"] = uniform(rng, nh * dh, (d, nh, dh))
        tensors[f"l{i}_ln1_g"] = np.ones(d, np.float32)
        tensors[f"l{i}_ln1_b"] = np.zeros(d, np.float32)
        tensors[f"l{i}_ln2_g"] = np.ones(d, np.float32)
        tensors[f"l{i}_ln2_b"] = np.zeros(d, np.float32)
        tensors[f"l{i}_ffn_w1"] = uniform(rng, d, (dff, d))
        tensors[f"l{i}_ffn_b1"] = np.zeros(dff, np.float32)
        tensors[f"l{i}_ffn_w2"] = uniform(rng, dff, (d, dff))
        tensors[f"l{i}_ffn_b2"] = np.zeros(d, np.float32)
    tensors["lnf_g"] = np.ones(d, np.float32)
    tensors["lnf_b"] = np.zeros(d, np.float32)
    tensors["ids"] = rng.integers(0, v, (total, seq)).astype(np.float32)
    tensors["tgt"] = rng.integers(0, v, (total, seq)).astype(np.float32)
    for key in ("n_layer", "n_head", "d_model", "d_ff", "vocab", "seq_len"):
        meta[key] = str(spec[key])


def fixture_path(out_dir: Path, name):
    """Where `build` writes the fixture for spec `name`: `out_dir/<name>.safetensors`, as a
    regular file, on every measuring host. So a name is a portable name
    (`fixture_digest.is_portable_name`, the rule origins and recorded fixture names are held to)
    rather than anything a path parser accepts: that excludes every separator, drive, `..` and
    whitespace (a name that would put the bytes outside `out_dir`, with `--out-dir` onto a
    recorded fixture without touching its digest), and a Windows device stem, which names no file
    in `out_dir` at all."""
    if not fixture_digest.is_portable_name(name):
        raise ValueError(f"spec name {name!r} is not a portable file name "
                         f"({fixture_digest.PORTABLE_NAME_RULE}): the fixture is written to "
                         f"{out_dir}/<name>.safetensors")
    return out_dir / f"{name}.safetensors"


def build(spec_path: Path, out_dir: Path):
    spec = json.loads(spec_path.read_text())
    out_path = fixture_path(out_dir, spec["name"])
    rng = np.random.default_rng(spec["seed"])
    model = spec.get("model", "mlp")
    tensors = {}
    meta = {
        "name": spec["name"],
        "model": model,
        "mode": spec.get("mode", "train"),
        "batch_size": str(spec["batch_size"]),
        "lr": repr(spec.get("lr", 0.0)),
        # Initial f16 loss scale for the mixed-precision legs (torch's GradScaler default). It is
        # a workload property: a scale whose first step already overflows costs the dynamic legs
        # backoff steps inside the parity window, and diverges the fixed-scale leg outright.
        "loss_scale": repr(float(spec.get("loss_scale", 65536.0))),
        "seed": str(spec["seed"]),
        "parity_steps": str(spec["parity_steps"]),
        "warmup_steps": str(spec["warmup_steps"]),
        "timed_steps": str(spec["timed_steps"]),
    }
    {"mlp": build_mlp, "conv": build_conv, "gpt": build_gpt}[model](spec, rng, tensors, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=meta)
    return out_path


def report_written(path):
    """The per-fixture line, printed by the caller once `path` is where the fixture stays."""
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return path


def build_smoke(specs, out_dir: Path):
    """Build `specs` into `out_dir`, each one replacing its destination ENTRY only once built.

    save_file writes THROUGH an existing entry, so building in place would let a symlink -- or a
    hard link -- in `out_dir` to a recorded fixture have the smoke bytes overwrite that fixture
    with its digest untouched. Each spec is built into a staging directory beside the
    destination and renamed over it: the rename replaces `out_dir`'s own directory entry and
    nothing it points to, and a spec that fails to build leaves its previous output (and every
    later spec's) where it was. The recording path (`build_recorded`) stages too, but replaces a
    symlink's TARGET rather than the entry: a symlinked recorded fixture is how measurement trees
    share ONE fixture file (gh612_cells.sh).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gen_fixtures-", dir=out_dir))
    written = []
    try:
        for spec in specs:
            staged = build(spec, staging)
            final = out_dir / staged.name
            os.replace(staged, final)
            written.append(report_written(final))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"recorded no digests: {len(written)} fixture(s) in {out_dir} are for smoke runs only")
    return written


def recorded_target(dest: Path):
    """The file the recording path replaces for the destination entry `dest`: the entry's
    resolved target, so a symlinked recorded fixture stays a symlink and every tree linking it
    sees the new bytes -- how measurement trees share ONE fixture file (gh612_cells.sh). Refuses
    a target the rename could not replace; `main` calls it for every spec before building any."""
    target = Path(os.path.realpath(dest))
    # Only a link can point into a missing directory: a plain entry's is `out_dir`, which a first
    # generation creates.
    if dest.is_symlink() and not target.parent.is_dir():
        raise ValueError(f"{dest} resolves to {target}, whose directory does not exist")
    if target.exists() and not target.is_file():
        raise ValueError(f"{dest} resolves to {target}, which is not a regular file")
    return target


def build_recorded(specs, out_dir: Path, commit):
    """Build `specs` for recording into `out_dir`, replacing nothing until ALL of them built.

    Staged (gh-ocannl-1059): a spec that fails to build -- no `seed`, an unknown `model`, a
    `build_conv` divisibility assertion, a disk-full `save_file` -- leaves every fixture in
    `out_dir`, the bytes the published numbers are on, exactly as it was, and nothing is
    recorded. Only then is each staged file renamed onto its destination's RESOLVED target rather
    than onto the entry: a symlinked recorded fixture stays a symlink whose shared target holds
    the new bytes. (A hard-linked one is un-shared, since a rename replaces one name; that is
    announced.) `commit(written, notes)` is then handed the replaced destination entries (what
    `record()` names) with the lines to print under each -- also when a rename fails or anything
    else interrupts the renames, for the fixtures replaced before it, so no replaced fixture is
    left unrecorded. Returns the destination entries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gen_fixtures-", dir=out_dir))
    sidecars = []
    try:
        staged = []
        for spec in specs:
            try:
                staged.append(build(spec, staging))
            except BaseException:
                print(f"gen_fixtures: FAILED to build {spec}; no fixture in {out_dir} was "
                      "replaced and no digest was recorded", file=sys.stderr)
                raise
        # Every staged file goes onto its target's filesystem before any is renamed: a symlink
        # into another filesystem needs a copy (os.replace cannot cross one), and that copy must
        # fail, if it fails, while nothing has been replaced yet.
        moves = []
        for path in staged:
            dest = out_dir / path.name
            target = recorded_target(dest)
            source = path
            if os.stat(path).st_dev != os.stat(target.parent).st_dev:
                fd, tmp = tempfile.mkstemp(prefix=".gen_fixtures-", suffix=".tmp",
                                           dir=target.parent)
                os.close(fd)
                source = Path(tmp)
                sidecars.append(source)
                shutil.copyfile(path, source)
                # mkstemp creates it owner-only; the fixture gets the mode build() gave it.
                shutil.copymode(path, source)
            moves.append((source, dest, target))
        # From the first rename on, the replaced fixtures are new bytes, so whatever ends the loop
        # -- a failed rename, an interrupt -- `commit` records them before the error propagates.
        # It records BEFORE it prints anything: an output that fails (a closed pipe) must not
        # cost the record either.
        written, notes, failed = [], {}, None
        try:
            for source, dest, target in moves:
                links = target.stat().st_nlink if target.exists() else 1
                failed = target
                os.replace(source, target)
                failed = None
                written.append(dest)
                notes[dest] = []
                if dest.is_symlink():
                    notes[dest].append(f"through the symlink to {target}, shared by every tree "
                                       "linking it")
                if links > 1:
                    notes[dest].append(f"{dest} had {links - 1} other hard link(s): they keep "
                                       "the previous bytes")
        finally:
            if written:
                commit(written, notes)
            if failed is not None:
                print(f"gen_fixtures: FAILED to replace {failed}; the {len(written)} fixture(s) "
                      "replaced before it were recorded", file=sys.stderr)
        return written
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        for sidecar in sidecars:
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass


def record_and_report(digests: Path, origin, written, notes):
    """Record `written` as `origin`'s bytes in `digests`, THEN report them: the record first, so a
    report that fails to print (a closed stdout pipe) cannot leave replaced fixtures unrecorded.
    `notes` maps a fixture to the lines printed under its `wrote` line."""
    # fixtures/ is gitignored, so this file is the only record of what was just generated
    # (gh-ocannl-645). Only this origin's regenerated entries are rewritten: generating one
    # workload must not drop the identities of the fixtures already on disk, and generating on
    # one box must not drop the identities another box's published numbers rest on
    # (gh-ocannl-759).
    changes = fixture_digest.record(digests, written, origin)
    for fixture in written:
        report_written(fixture)
        for note in notes.get(fixture, ()):
            print(f"  {note}")
    print(f"recorded {len(written)} digest(s) in {digests} as origin {origin!r}")
    by_name = {fixture.name: fixture for fixture in written}
    for name, org, was, now in changes:
        replacement = fixture_digest.replacement_kind(by_name[name], was)
        if was is None:
            print(f"  new: {name} sha256 {now.sha256} [{org}]")
        elif replacement == "same-file-migration":
            print(f"  MIGRATED for {org}: {name} raw-v1 -> content-v1 (same file bytes)")
            print(f"    now  sha256 {now.sha256} ({now.size} bytes)")
        elif replacement == "new-content-baseline":
            print(f"  NEW CONTENT BASELINE for {org}: {name}")
            print(f"    historical raw-v1   sha256 {was.sha256} ({was.size} bytes)")
            print(f"    baseline content-v1 sha256 {now.sha256} ({now.size} bytes)")
        else:
            # Loudly, and not only in a git diff: numbers measured on the old bytes are not
            # comparable with numbers measured on the new ones, whatever the report calls the
            # workload.
            print(f"  CHANGED for {org}: {name}")
            print(f"    was  sha256 {was.sha256} ({was.size} bytes)")
            print(f"    now  sha256 {now.sha256} ({now.size} bytes)")
    replacements = {
        fixture_digest.replacement_kind(by_name[name], was) for name, _, was, _ in changes
    }
    if "content-change" in replacements:
        print("  a changed fixture is a changed workload: reports measured on the previous "
              "digest are not comparable with reports measured on this one.")
    if "new-content-baseline" in replacements:
        print("  the new content baseline is reproducible going forward, but does not prove "
              "continuity with the historical raw digest; old reports retain that digest.")
    others = fixture_digest.divergent_origins(digests, [fx.name for fx in written], origin)
    if others:
        # The trap gh-ocannl-759 was filed about: their entries survive (so their fixtures still
        # pass the gate), but their bytes are now a different workload from this box's. Only
        # A declared origin with no entry is named too: the declaration is what distinguishes
        # "this measuring box has unrecorded bytes" from "this box never measures this workload".
        # A coordinated regeneration that recorded the same bytes on both boxes remains quiet.
        print(f"  measurement boxes missing an entry or still on DIFFERENT bytes for these "
              f"fixtures: {', '.join(others)} — regeneration is a cross-box event, so until "
              "they regenerate and record too, their published numbers and this box's may be on "
              "different workloads.")


def main(argv=None, here=None):
    """Generate the requested fixtures and record their digests as this box's bytes.

    A function, not a bare `__main__` block, so the order of its steps is testable: what a
    fixture generator must never do is overwrite bytes it then turns out to be unable to record.
    `here` is the benchmarks directory (its `workloads/` and `fixtures/`), overridable for that.
    A spec that fails to build replaces no fixture and records nothing: the error propagates
    (a nonzero exit), naming the spec. With `--out-dir` it records nothing and touches neither
    `fixtures/` nor its digest file. Returns the paths written.
    """
    here = Path(__file__).parent if here is None else Path(here)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Regeneration is a CROSS-BOX event (gh-ocannl-759): the bytes depend on this "
        "box's numpy, so regenerating here does not give the other measuring boxes the same "
        "workload, and every number they published stays on their own bytes until they "
        "regenerate too. To merely pin fixtures that already exist, without changing any "
        f"workload, use `python3 {fixture_digest.cli_command()} --record` instead.",
    )
    ap.add_argument("specs", nargs="*", type=Path, help="workload specs (default: all of them)")
    ap.add_argument(
        "--origin",
        default=None,
        help="the box these bytes are, recorded with them "
        f"({fixture_digest.origin_default_help()})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write the fixtures into this directory instead, and record NO digest: bytes "
        "generated for a smoke run publish nothing, so they must not become an origin's entry in "
        "the tracked fixtures/DIGESTS.txt. orchestrate.py never sees them; hand one to a runner "
        "directly (BENCH_FIXTURE=<path>, --fixture <path>), for smoke runs only",
    )
    args = ap.parse_args(argv)
    specs = args.specs or sorted((here / "workloads").glob("*.json"))
    recording = args.out_dir is None
    if not recording:
        if args.origin is not None:
            ap.error("--origin names the box recorded bytes belong to; --out-dir records none")
        # The tracked directory is the one place these bytes must not land: writing there without
        # recording overwrites the fixtures the published numbers are on and leaves the new bytes
        # matching no entry -- the loss the digest validation below exists to prevent. By
        # filesystem identity, not by path: a case alias on a case-insensitive filesystem, a
        # symlink or a `..` detour all name the same directory under different spellings. And
        # AFTER creating DIR, since creation can turn a spelling into an alias: `fixtures/new/..`
        # names nothing until `new` exists, and fixtures/ itself from then on.
        fixtures = here / "fixtures"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if fixtures.exists() and os.path.samefile(args.out_dir, fixtures):
            ap.error(f"--out-dir {args.out_dir} is the recorded fixtures directory; "
                     "regenerate there without --out-dir, so the digests are recorded")
        out_dir = args.out_dir
    else:
        # BEFORE building anything: generating rewrites the fixture bytes, so a bad origin
        # discovered at the recording step would leave a regenerated workload that cannot be
        # attributed. Through resolve_origin, which refuses an explicitly empty value instead of
        # substituting this host.
        origin = fixture_digest.resolve_origin(args.origin)
        out_dir = here / "fixtures"
        digests = out_dir / fixture_digest.DIGEST_FILE
        # And for the same reason, parse the digest file BEFORE building: once every spec has
        # built, its bytes REPLACE the fixtures, so anything record() would refuse -- a pre-gh-ocannl-759 three-field
        # line, a duplicate origin, a malformed row -- has to be discovered while the previous
        # bytes still exist. Refusing afterwards leaves regenerated bytes that nothing records AND
        # the bytes the published numbers were measured on gone, which is worse than either alone.
        fixture_digest.read_digests(digests)
    # Names too, and for the same reason: build() writes <out_dir>/<spec name>.safetensors, so a
    # name with a path component would write outside out_dir (with --out-dir, onto a recorded
    # fixture), and a name the digest format cannot carry would be refused by record() only AFTER
    # the previous bytes are overwritten. fixture_path holds names to the one rule record()'s
    # check_fixture_name applies, so it refuses both; build() re-checks it itself, and checking
    # every spec here refuses before ANY is built. A spec this cannot parse is left for build() to
    # refuse on its own terms -- like any failure inside build(), before anything is replaced:
    # every spec builds into a staging directory first (gh-ocannl-1059). Two specs with
    # one destination are refused as well: the later would silently replace the earlier's
    # fixture. Compared case-folded (names are ASCII, by fixture_path), because on the
    # case-insensitive macOS and Windows measuring hosts `Lenet` and `lenet` are one file.
    claimed, targets = {}, {}
    for spec_path in specs:
        try:
            name = json.loads(spec_path.read_text())["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        fixture_path(out_dir, name)
        if name.lower() in claimed:
            raise ValueError(f"{claimed[name.lower()]} and {spec_path} both generate "
                             f"{name}.safetensors (names are compared ignoring case); the later "
                             "would replace the earlier")
        claimed[name.lower()] = spec_path
        if recording:
            # The recording path replaces a symlinked fixture's TARGET, so the destination that
            # must be unique and replaceable is the resolved one: two fixtures linked to one file
            # would have the later spec's bytes recorded under both names.
            target = recorded_target(fixture_path(out_dir, name))
            key = str(target).lower()
            if key in targets:
                raise ValueError(f"{targets[key]} and {spec_path} both generate {target} "
                                 "(through a symlink; compared ignoring case); the later would "
                                 "replace the earlier")
            targets[key] = spec_path
    if not recording:
        return build_smoke(specs, out_dir)
    return build_recorded(
        specs, out_dir, lambda written, notes: record_and_report(digests, origin, written, notes))


if __name__ == "__main__":
    main()
