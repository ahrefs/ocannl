#!/usr/bin/env python3
"""tinygrad runner for the cross-framework benchmark suite (see benchmarks/README.md).

Model-dispatched on the fixture metadata: mlp / conv (LeNet-5, valid convs) / gpt
(pre-LN GPT-2-style decoder, trained or inference-only per the fixture's `mode`). Math
mirrors the OCANNL and PyTorch runners exactly; see the PyTorch runner's docstring for the
conventions.
"""

import argparse
import math
import os
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", required=True)
# AMD is tinygrad's Linux ROCm device; CL (OpenCL) reaches AMD GPUs on Windows; HIP goes through
# the HIP runtime, which is what works under WSL (no /dev/kfd for AMD to open). orchestrate --gpu
# hip maps to AMD on Linux, HIP when /dev/kfd is absent, and CL elsewhere.
ap.add_argument("--device", default="CPU", choices=["CPU", "METAL", "CUDA", "AMD", "CL", "HIP"])
ap.add_argument("--jit", type=int, default=1)
ap.add_argument("--beam", type=int, default=0, help="BEAM search width (0 = off); implies --jit")
# gh-ocannl-675 probe: time a SECOND block of timed_steps inside the same process, after the
# queued block, to separate "a process that searched is slower per launch" from first-block
# warmup. Off by default; the published cell shape is unchanged.
ap.add_argument("--retime", action="store_true", help="time a second block of steps (gh-675)")
args = ap.parse_args()
os.environ["DEV"] = args.device
if args.beam:
    # Must be set before the tinygrad import: BEAM is read into a ContextVar at import time.
    os.environ["BEAM"] = str(args.beam)
    args.jit = 1

# BEFORE importing tinygrad: this file lives in runners/tinygrad, so any `runners/` entry on
# sys.path -- including one inherited from PYTHONPATH, which nothing here put there -- makes the
# scan resolve `tinygrad` to THIS directory as a namespace-package portion and the import fails
# outright with `cannot import name 'Tensor' from 'tinygrad' (unknown location)`. Purge every
# equivalent entry first; the block after the imports puts one back briefly for `bench_common`.
_RUNNERS = str(Path(__file__).resolve().parent.parent)


def _drop_runners_from_path():
    sys.path[:] = [q for q in sys.path if not q or Path(q).resolve() != Path(_RUNNERS)]


_drop_runners_from_path()

import numpy as np
from safetensors.numpy import load_file
from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import SGD

# After the tinygrad import: this directory is runners/tinygrad, so with runners/ on
# sys.path an `import tinygrad` scan would first hit it as a namespace-package portion —
# which shadows editable (finder-based) tinygrad installs, whose MetaPath finder is only
# consulted when the path scan finds nothing.
sys.path.insert(0, _RUNNERS)
from bench_common import (
    emit,
    instrument_tinygrad_beam,
    percentiles,
    read_st_metadata,
    tinygrad_searched,
)

# ...and off again -- EVERY equivalent entry, not just the one inserted above: the runner can
# be launched with `benchmarks/runners` already on PYTHONPATH, and removing a single copy
# would leave the inherited one behind, which is enough to reproduce the whole failure.
# tinygrad's beam search runs its candidate compiles in a `spawn` pool, and a
# spawned worker re-executes THIS module top-level with the parent's sys.path — where the
# `import tinygrad` above happens before the insert. With runners/ still on the path the scan
# finds runners/tinygrad/ as a namespace portion first, the editable install's meta-path finder
# (appended after PathFinder) never gets a look, and every worker dies with `cannot import name
# 'Tensor' from 'tinygrad' (unknown location)`. The pool respawns them forever, so the search
# wedges instead of failing (gh-ocannl-675 CUDA leg).
_drop_runners_from_path()


def param(arr):
    t = Tensor(arr)
    t.requires_grad = True
    return t


def ce_onehot(logits, y_onehot):
    probs = logits.softmax(-1)
    return -((probs * y_onehot).sum(-1)).log().mean()


def build_mlp(meta, data):
    n_layers = int(meta["n_layers"])
    params = [(param(data[f"w{i}"]), param(data[f"b{i}"])) for i in range(1, n_layers + 1)]
    flat = [p for wb in params for p in wb]

    def forward(xb):
        h = xb
        for i, (w, b) in enumerate(params):
            h = h.linear(w.T, b)
            if i < n_layers - 1:
                h = h.relu()
        return h

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    return loss_fn, flat, (data["x"], data["y"]), None


def build_conv(meta, data):
    w1 = param(np.ascontiguousarray(data["conv1_kernel"].transpose(0, 3, 1, 2)))
    b1 = param(data["conv1_bias"])  # per-channel bias [oc]
    w2 = param(np.ascontiguousarray(data["conv2_kernel"].transpose(0, 3, 1, 2)))
    b2 = param(data["conv2_bias"])
    wf1 = param(data["fc1_w"].reshape(data["fc1_w"].shape[0], -1))
    bf1 = param(data["fc1_b"])
    wf2 = param(data["fc2_w"])
    bf2 = param(data["fc2_b"])
    wl = param(data["w_logits"])
    bl = param(data["b_logits"])
    flat = [w1, b1, w2, b2, wf1, bf1, wf2, bf2, wl, bl]

    # Same-padding (cifar-scale workload) vs valid (LeNet); odd kernel keeps the spatial extent.
    # Strides (cifar_stride's stride-2 stem, gh-ocannl-502) are valid-only — see gen_fixtures.
    k = int(meta.get("kernel_size", "5"))
    pad = (k - 1) // 2 if meta.get("use_padding", "false") == "true" else 0
    s1, s2 = int(meta.get("stride1", "1")), int(meta.get("stride2", "1"))

    def forward(xb):
        z = xb.conv2d(w1, b1, stride=s1, padding=pad)
        z = z.relu().max_pool2d(kernel_size=(2, 2))
        z = z.conv2d(w2, b2, stride=s2, padding=pad)
        z = z.relu().max_pool2d(kernel_size=(2, 2))
        h = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1)
        h = h.linear(wf1.T, bf1).relu()
        h = h.linear(wf2.T, bf2).relu()
        return h.linear(wl.T, bl)

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    x = np.ascontiguousarray(data["x"].transpose(0, 3, 1, 2))  # NCHW
    return loss_fn, flat, (x, data["y"]), None


def gelu_tanh(x):
    return 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())


def layernorm(x, g, b):
    mu = x.mean(-1, keepdim=True)
    c = x - mu
    var = (c * c).mean(-1, keepdim=True)
    return c / (var + 1e-5).sqrt() * g + b


def build_gpt(meta, data):
    n_layer, nh = int(meta["n_layer"]), int(meta["n_head"])
    d, v, seq = int(meta["d_model"]), int(meta["vocab"]), int(meta["seq_len"])
    dh = d // nh
    # mode: train (gpt2_mini_train) makes every weight a parameter of the SGD step; mode: infer
    # (gpt2_mini) keeps them plain constants, as before.
    training = meta.get("mode", "train") == "train"
    leaf = param if training else Tensor
    wte = leaf(data["wte"])  # [d, v]
    wpe = leaf(data["wpe"])
    layers = [
        {
            "wq": leaf(data[f"l{i}_wq"].reshape(nh * dh, d)),
            "wk": leaf(data[f"l{i}_wk"].reshape(nh * dh, d)),
            "wv": leaf(data[f"l{i}_wv"].reshape(nh * dh, d)),
            "wo": leaf(data[f"l{i}_wo"].reshape(d, nh * dh)),
            "g1": leaf(data[f"l{i}_ln1_g"]),
            "b1": leaf(data[f"l{i}_ln1_b"]),
            "g2": leaf(data[f"l{i}_ln2_g"]),
            "b2": leaf(data[f"l{i}_ln2_b"]),
            "fw1": leaf(data[f"l{i}_ffn_w1"]),
            "fb1": leaf(data[f"l{i}_ffn_b1"]),
            "fw2": leaf(data[f"l{i}_ffn_w2"]),
            "fb2": leaf(data[f"l{i}_ffn_b2"]),
        }
        for i in range(n_layer)
    ]
    gf, bf = leaf(data["lnf_g"]), leaf(data["lnf_b"])
    flat = (
        [wte, wpe, gf, bf] + [p for layer in layers for p in layer.values()] if training else []
    )
    mask = Tensor(np.tril(np.ones((seq, seq), np.bool_)))  # [s, t]
    # The embedding table is the transpose of the tied lm_head weight; under training it is
    # recomputed inside forward so its gradient reaches wte, instead of being realized once.
    emb_once = None if training else wte.T.contiguous().realize()  # [v, d]

    def forward(ids_onehot):
        b = ids_onehot.shape[0]
        emb = wte.T if training else emb_once
        x = ids_onehot @ emb + wpe  # [b,s,d]
        for p in layers:
            h = layernorm(x, p["g1"], p["b1"])
            q = (h @ p["wq"].T).reshape(b, seq, nh, dh)
            k = (h @ p["wk"].T).reshape(b, seq, nh, dh)
            vv = (h @ p["wv"].T).reshape(b, seq, nh, dh)
            att = Tensor.einsum("bshd,bthd->bsth", q, k) / math.sqrt(dh)
            att = mask.reshape(1, seq, seq, 1).where(att, -1e9)
            att = att.softmax(2)
            out = Tensor.einsum("bsth,bthe->bshe", att, vv).reshape(b, seq, nh * dh)
            x = x + out @ p["wo"].T
            h2 = layernorm(x, p["g2"], p["b2"])
            x = x + (gelu_tanh(h2 @ p["fw1"].T + p["fb1"]) @ p["fw2"].T + p["fb2"])
        x = layernorm(x, gf, bf)
        return x @ wte  # tied lm_head

    def loss_fn(ids_onehot, tgt_onehot):
        logp = forward(ids_onehot).log_softmax(-1)
        return -((logp * tgt_onehot).sum(-1)).mean()

    def one_hot(a):
        out = np.zeros((*a.shape, v), np.float32)
        np.put_along_axis(out, a[..., None].astype(np.int64), 1.0, axis=-1)
        return out

    ids = one_hot(data["ids"])
    tgt = one_hot(data["tgt"])
    return loss_fn, flat, (ids, tgt), int(meta["batch_size"]) * seq


def main():
    meta = read_st_metadata(args.fixture)
    model = meta.get("model", "mlp")
    mode = meta.get("mode", "train")
    batch_size = int(meta["batch_size"])
    lr = float(meta["lr"])
    parity_steps = int(meta["parity_steps"])
    warmup_steps = int(meta["warmup_steps"])
    timed_steps = int(meta["timed_steps"])

    data = load_file(args.fixture)
    build = {"mlp": build_mlp, "conv": build_conv, "gpt": build_gpt}[model]
    loss_fn, flat, (x, y), tokens_per_step = build(meta, data)
    n_batches = x.shape[0] // batch_size
    batches = [
        (
            Tensor(np.ascontiguousarray(x[i * batch_size : (i + 1) * batch_size])).realize(),
            Tensor(np.ascontiguousarray(y[i * batch_size : (i + 1) * batch_size])).realize(),
        )
        for i in range(n_batches)
    ]

    if mode == "train":
        opt = SGD(flat, lr=lr)

        def step_inner(xb, yb):
            loss = loss_fn(xb, yb)
            opt.zero_grad()
            loss.backward()
            # Realize the loss value before opt.step(): the step assigns params in place, and
            # a later realize would recompute the (fused-away) loss from the updated weights.
            loss.realize()
            opt.step()
            return loss

    else:

        def step_inner(xb, yb):
            return loss_fn(xb, yb).realize()

    if args.jit:
        step_inner = TinyJit(step_inner)

    def step(k):
        xb, yb = batches[k % n_batches]
        return step_inner(xb, yb)

    # Before the first launch, which is where the beam search happens: whether this process
    # searched or replayed ~/.cache/tinygrad is what the result line's `searched` reports
    # (gh-ocannl-644). Unlike an OCANNL tuned cell, a beam cell searches in the process that
    # then times steps — the report says so rather than leaving it to be assumed either way
    # (gh-ocannl-675).
    beam_counts = instrument_tinygrad_beam() if args.beam else None

    def sync():
        from tinygrad import Device

        Device[Device.DEFAULT].synchronize()

    # Tensor.train was replaced on tinygrad master by the TRAINING context var (the optimizer
    # refuses to step outside it).
    if hasattr(Tensor, "train"):
        train_ctx = Tensor.train(mode == "train")
    else:
        from tinygrad.helpers import Context

        train_ctx = Context(TRAINING=int(mode == "train"))
    with train_ctx:
        k = 0
        losses = []
        t0 = time.perf_counter()
        losses.append(step(k).item())
        compile_s = time.perf_counter() - t0
        k += 1
        for _ in range(parity_steps - 1):
            losses.append(step(k).item())
            k += 1
        for _ in range(warmup_steps):
            step(k)
            k += 1
        sync()
        synced = []
        for _ in range(timed_steps):
            t0 = time.perf_counter()
            step(k)
            k += 1
            sync()
            synced.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        for _ in range(timed_steps):
            step(k)
            k += 1
        sync()
        queued = (time.perf_counter() - t0) / timed_steps * 1e3
        retimed = None
        if args.retime:
            sync()
            retimed = []
            for _ in range(timed_steps):
                t0 = time.perf_counter()
                step(k)
                k += 1
                sync()
                retimed.append((time.perf_counter() - t0) * 1e3)

    result = {
        "framework": "tinygrad",
        "backend": args.device,
        "variant": "beam" if args.beam else ("jit" if args.jit else "nojit"),
        "workload": meta["name"],
        "compile_s": round(compile_s, 3),
        "searched": tinygrad_searched(beam_counts, args.beam),
        # No exact arm exists for tinygrad: its default reassociates freely, so the sweep stands
        # this one cell in the approximate regime whenever it runs one (gh-ocannl-719).
        "regime_settings": "tinygrad defaults (no exact pin; reassociates freely)",
        "step_ms": percentiles(synced),
        "queued_step_ms": queued,
        "timed_steps": timed_steps,
        "losses": losses,
        "version": pkg_version("tinygrad"),
    }
    if retimed:
        result["retime_step_ms"] = percentiles(retimed)
    if tokens_per_step:
        result["tokens_per_step"] = tokens_per_step
    emit(result)


if __name__ == "__main__":
    main()
