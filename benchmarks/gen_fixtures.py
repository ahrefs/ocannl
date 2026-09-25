#!/usr/bin/env python3
"""Generate self-describing safetensors fixtures for the cross-framework benchmark suite.

Each fixture holds the initial weights, the full dataset (inputs and one-hot labels), and
the workload hyperparameters in the safetensors __metadata__ map, so every runner needs
only the fixture path. All payloads are float32; weights are [fan_out, fan_in] row-major
(the shared convention: PyTorch nn.Linear layout, OCANNL output-axes-then-input-axes).
"""

import argparse
import json
import re
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


#: Windows device names, reserved whatever extension follows them (`CON.safetensors` is CON).
WINDOWS_DEVICES = {
    "CON", "PRN", "AUX", "NUL", *(f"{dev}{i}" for dev in ("COM", "LPT") for i in range(1, 10))
}


def fixture_path(out_dir: Path, name):
    """Where `build` writes the fixture for spec `name`: `out_dir/<name>.safetensors`, as a
    regular file, on every measuring host. So a name is an allowlisted portable word -- the
    alphabet `fixture_digest.check_origin` holds origins to -- rather than anything a path parser
    accepts: that excludes every separator, drive, `..` and whitespace (a name that would put the
    bytes outside `out_dir`, with `--out-dir` onto a recorded fixture without touching its digest),
    and a Windows device stem, which names no file in `out_dir` at all."""
    if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or name.split(".")[0].upper() in WINDOWS_DEVICES):
        raise ValueError(f"spec name {name!r} is not a portable file name (an ASCII letter or "
                         "digit, then letters, digits, dot, underscore or hyphen; no Windows "
                         f"device name): the fixture is written to {out_dir}/<name>.safetensors")
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
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def main(argv=None, here=None):
    """Generate the requested fixtures and record their digests as this box's bytes.

    A function, not a bare `__main__` block, so the order of its steps is testable: what a
    fixture generator must never do is overwrite bytes it then turns out to be unable to record.
    `here` is the benchmarks directory (its `workloads/` and `fixtures/`), overridable for that.
    With `--out-dir` it records nothing and touches neither `fixtures/` nor its digest file.
    Returns the paths written.
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
        # matching no entry -- the loss the digest validation below exists to prevent.
        if args.out_dir.resolve() == (here / "fixtures").resolve():
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
        # And for the same reason, parse the digest file BEFORE building: building OVERWRITES the
        # fixture bytes, so anything record() would refuse -- a pre-gh-ocannl-759 three-field
        # line, a duplicate origin, a malformed row -- has to be discovered while the previous
        # bytes still exist. Refusing afterwards leaves regenerated bytes that nothing records AND
        # the bytes the published numbers were measured on gone, which is worse than either alone.
        fixture_digest.read_digests(digests)
    # Names too, and for the same reason: build() writes <out_dir>/<spec name>.safetensors, so a
    # name with a path component would write outside out_dir (with --out-dir, onto a recorded
    # fixture), and a name the digest format cannot carry would be refused by record() only AFTER
    # the previous bytes are overwritten. build() re-checks the first itself; checking every spec
    # here refuses before ANY is built. A spec this cannot parse is left for build() to refuse on
    # its own terms -- that refusal also happens before that spec mutates anything.
    destinations = []
    for spec_path in specs:
        try:
            name = json.loads(spec_path.read_text())["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        destinations.append(fixture_path(out_dir, name))
        if recording:
            fixture_digest.check_fixture_name(f"{name}.safetensors")
    if not recording:
        # save_file writes THROUGH an existing entry: a symlink in DIR (to a recorded fixture, say)
        # or a hard link to one would have the smoke bytes overwrite that fixture with its digest
        # untouched. Unlinking removes DIR's own directory entry and nothing it points to, so the
        # build then creates a fresh regular file. After every name has passed, so a refusal
        # above leaves DIR as it was.
        for path in destinations:
            if path.is_symlink() or path.is_file():
                path.unlink()
    written = [build(spec, out_dir) for spec in specs]
    if not recording:
        print(f"recorded no digests: {len(written)} fixture(s) in {out_dir} are for smoke runs "
              "only")
        return written
    # fixtures/ is gitignored, so this file is the only record of what was just generated
    # (gh-ocannl-645). Only this origin's regenerated entries are rewritten: generating one
    # workload must not drop the identities of the fixtures already on disk, and generating on
    # one box must not drop the identities another box's published numbers rest on
    # (gh-ocannl-759).
    changes = fixture_digest.record(digests, written, origin)
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
    return written


if __name__ == "__main__":
    main()
