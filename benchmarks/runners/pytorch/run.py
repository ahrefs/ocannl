#!/usr/bin/env python3
"""PyTorch runner for the cross-framework benchmark suite (see benchmarks/README.md).

Model-dispatched on the fixture metadata: mlp / conv (LeNet-5, valid convs) / gpt
(pre-LN GPT-2-style decoder, trained or inference-only per the fixture's `mode`).
Math mirrors the OCANNL runners exactly:
same CE formulation (softmax -> one-hot dot -> log), same LayerNorm (biased variance,
eps inside sqrt), same tanh-gelu constant, same conv layouts (fixture is channels-last /
OCANNL axis order; this runner permutes to NCHW once). The causal mask fills with -1e9
here and with -inf in OCANNL since gh-ocannl-548 — the same value after exp at every
precision these runners use.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import (
    emit,
    percentiles,
    read_st_metadata,
    torch_searched,
)

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def ce_onehot(logits, y_onehot):
    probs = torch.softmax(logits, dim=-1)
    correct = (probs * y_onehot).sum(dim=-1)
    return (-correct.log()).mean()


def build_mlp(meta, data, dev, approximate=False):
    n_layers = int(meta["n_layers"])
    params = []
    for i in range(1, n_layers + 1):
        params.append(
            (data[f"w{i}"].to(dev).requires_grad_(), data[f"b{i}"].to(dev).requires_grad_())
        )
    flat = [p for wb in params for p in wb]

    def forward(xb):
        h = xb
        for i, (w, b) in enumerate(params):
            h = h @ w.T + b
            if i < n_layers - 1:
                h = torch.relu(h)
        return h

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    x, y = data["x"].to(dev), data["y"].to(dev)
    return loss_fn, flat, (x, y), None


def build_conv(meta, data, dev, approximate=False):
    def leaf(t):
        return t.contiguous().to(dev).requires_grad_()

    w1 = leaf(data["conv1_kernel"].permute(0, 3, 1, 2))  # [oc,ic,kh,kw]
    b1 = leaf(data["conv1_bias"])  # per-channel bias [oc]
    w2 = leaf(data["conv2_kernel"].permute(0, 3, 1, 2))
    b2 = leaf(data["conv2_bias"])
    wf1 = leaf(data["fc1_w"].reshape(data["fc1_w"].shape[0], -1))  # [hid, oh*ow*oc]
    bf1 = leaf(data["fc1_b"])
    wf2 = leaf(data["fc2_w"])
    bf2 = leaf(data["fc2_b"])
    wl = leaf(data["w_logits"])
    bl = leaf(data["b_logits"])
    flat = [w1, b1, w2, b2, wf1, bf1, wf2, bf2, wl, bl]

    # Same-padding (cifar-scale workload) vs valid (LeNet); odd kernel keeps the spatial extent.
    # Strides (cifar_stride's stride-2 stem, gh-ocannl-502) are valid-only — see gen_fixtures.
    k = int(meta.get("kernel_size", "5"))
    pad = (k - 1) // 2 if meta.get("use_padding", "false") == "true" else 0
    s1, s2 = int(meta.get("stride1", "1")), int(meta.get("stride2", "1"))

    def forward(xb):
        z = F.conv2d(xb, w1, b1, stride=s1, padding=pad)
        z = F.max_pool2d(torch.relu(z), 2)
        z = F.conv2d(z, w2, b2, stride=s2, padding=pad)
        z = F.max_pool2d(torch.relu(z), 2)
        h = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1)  # match OCANNL [oh,ow,oc] order
        h = torch.relu(h @ wf1.T + bf1)
        h = torch.relu(h @ wf2.T + bf2)
        return h @ wl.T + bl

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    x = data["x"].permute(0, 3, 1, 2).contiguous().to(dev)  # NCHW
    y = data["y"].to(dev)
    return loss_fn, flat, (x, y), None


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def layernorm(x, g, b):
    mu = x.mean(-1, keepdim=True)
    c = x - mu
    var = (c * c).mean(-1, keepdim=True)
    return c / torch.sqrt(var + 1e-5) * g + b


def build_gpt(meta, data, dev, approximate=False):
    n_layer, nh = int(meta["n_layer"]), int(meta["n_head"])
    d, v, seq = int(meta["d_model"]), int(meta["vocab"]), int(meta["seq_len"])
    dh = d // nh
    # mode: train (gpt2_mini_train) makes every weight a leaf parameter of the plain-SGD step;
    # mode: infer (gpt2_mini) keeps them plain constants, as before.
    training = meta.get("mode", "train") == "train"

    def leaf(t):
        t = t.contiguous().to(dev)
        return t.requires_grad_() if training else t

    wte = leaf(data["wte"])  # [d, v]
    wpe = leaf(data["wpe"])
    layers = []
    for i in range(n_layer):
        layers.append(
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
        )
    gf, bf = leaf(data["lnf_g"]), leaf(data["lnf_b"])
    flat = (
        [wte, wpe, gf, bf] + [p for layer in layers for p in layer.values()] if training else []
    )
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=dev))  # [s, t]
    # The embedding table is the transpose of the tied lm_head weight. Under training it must be
    # rebuilt inside forward: built once, its autograd graph would be freed by the first backward
    # and the second step would fail. Inference keeps the one-time transpose. Keep the
    # `.contiguous()` in both: gathering rows out of the transposed *view* takes a slow path here
    # (measured 5.4 s/step vs 98 ms on the gpt2_mini_train fixture, CPU) — a runner artifact that
    # would misreport the reference, not a property of the model.
    emb_once = None if training else wte.T.contiguous()  # [v, d]

    def forward(ids):
        b = ids.shape[0]
        emb = wte.T.contiguous() if training else emb_once
        x = emb[ids] + wpe
        for p in layers:
            h = layernorm(x, p["g1"], p["b1"])
            q = (h @ p["wq"].T).view(b, seq, nh, dh)
            k = (h @ p["wk"].T).view(b, seq, nh, dh)
            vv = (h @ p["wv"].T).view(b, seq, nh, dh)
            if approximate:
                # gh-ocannl-719: the approximate arm uses torch's fused attention (flash /
                # memory-efficient / math, its own choice), causal by flag, in its [b, nh, s, dh]
                # layout -- the counterpart of OCANNL's online-softmax gate once that lands, and
                # what a PyTorch user actually runs. The composed form below is the exact arm's.
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2), is_causal=True
                )
                out = out.transpose(1, 2).reshape(b, seq, nh * dh)
            else:
                att = torch.einsum("bshd,bthd->bsth", q, k) / math.sqrt(dh)
                att = torch.where(mask[None, :, :, None], att, torch.tensor(-1e9, device=dev))
                att = torch.softmax(att, dim=2)
                out = torch.einsum("bsth,bthe->bshe", att, vv).reshape(b, seq, nh * dh)
            x = x + out @ p["wo"].T
            h2 = layernorm(x, p["g2"], p["b2"])
            x = x + (gelu_tanh(h2 @ p["fw1"].T + p["fb1"]) @ p["fw2"].T + p["fb2"])
        x = layernorm(x, gf, bf)
        return x @ wte  # tied lm_head

    def loss_fn(ids, tgt_onehot):
        logp = torch.log_softmax(forward(ids), dim=-1)
        return -((logp * tgt_onehot).sum(-1)).mean()

    ids = data["ids"].long().to(dev)
    tgt = F.one_hot(data["tgt"].long(), v).float().to(dev)
    return loss_fn, flat, (ids, tgt), int(meta["batch_size"]) * seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--compile", action="store_true")
    # gh-ocannl-675: max-autotune is the honest analogue of a tuned cell (it benchmarks kernels).
    # A separate variant, not a change to the `compiled` cell.
    ap.add_argument("--compile-mode", default=None,
                    help="torch.compile mode, e.g. max-autotune; implies --compile (gh-675 probe)")
    ap.add_argument("--retime", action="store_true", help="time a second block of steps (gh-675)")
    # gh-ocannl-719: the exact arm pins what a parity oracle needs pinned (`highest` matmul
    # precision, no cudnn tf32, hand-composed attention). The approximate arm is torch's own
    # defaults -- what a PyTorch user actually runs with, and the fair counterpart of OCANNL's
    # `approximate` profile (the maintainer's decision on the issue) -- recorded per result line.
    ap.add_argument("--regime", default="exact", choices=["exact", "approximate"],
                    help="exact: matmul precision `highest`, cudnn tf32 off, composed attention; "
                    "approximate: `high` (tf32 where the device has it), cudnn.benchmark, "
                    "scaled_dot_product_attention (gh-ocannl-719)")
    args = ap.parse_args()
    approximate = args.regime == "approximate"
    # A mode is a torch.compile setting and means nothing without it: taking it alone would run
    # EAGER while stamping the result line with a `compile_mode`, i.e. a measurement labelled as
    # something it is not.
    if args.compile_mode:
        args.compile = True

    meta = read_st_metadata(args.fixture)
    model = meta.get("model", "mlp")
    mode = meta.get("mode", "train")
    batch_size = int(meta["batch_size"])
    lr = float(meta["lr"])
    parity_steps = int(meta["parity_steps"])
    warmup_steps = int(meta["warmup_steps"])
    timed_steps = int(meta["timed_steps"])

    torch.set_float32_matmul_precision("high" if approximate else "highest")
    torch.backends.cudnn.benchmark = approximate
    # cudnn's tf32 defaults ON, unlike matmul's: pinning it off is part of the exact arm's contract
    # on an Ampere-or-later device, and leaving the default is part of the approximate arm's.
    torch.backends.cudnn.allow_tf32 = approximate
    # What torch will EFFECTIVELY do, read back rather than assumed (Codex P1 on PR #661): an
    # ambient TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 forces tf32 matmuls whatever the precision set
    # above, and torch's own getter reflects the override. The regime this process reports is
    # derived from those effective settings, so an exact cell that torch quietly made approximate
    # is reported as approximate and the sweep's regime gate refuses the row -- the class, not the
    # one variable: anything a torch getter reflects is caught the same way.
    effective = {
        "matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "sdpa": approximate,
    }
    runner_regime = "exact" if not any(effective.values()) else "approximate"
    tf32_override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    regime_settings = (
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}, "
        + ", ".join(f"{k}={v}" for k, v in effective.items())
        + f", attention={'sdpa' if approximate else 'composed'}"
        + (f", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={tf32_override}" if tf32_override else "")
    )
    dev = torch.device(args.device)
    data = load_file(args.fixture)
    build = {"mlp": build_mlp, "conv": build_conv, "gpt": build_gpt}[model]
    loss_fn, flat, (x, y), tokens_per_step = build(meta, data, dev, approximate=approximate)
    n_batches = x.shape[0] // batch_size
    batches = [
        (x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size])
        for i in range(n_batches)
    ]

    if args.compile:
        loss_fn = (
            torch.compile(loss_fn, mode=args.compile_mode)
            if args.compile_mode
            else torch.compile(loss_fn)
        )

    if mode == "train":

        def step(k):
            xb, yb = batches[k % n_batches]
            loss = loss_fn(xb, yb)
            loss.backward()
            with torch.no_grad():
                for p in flat:
                    p -= lr * p.grad
            for p in flat:
                p.grad = None
            return loss

    else:

        def step(k):
            xb, yb = batches[k % n_batches]
            with torch.no_grad():
                return loss_fn(xb, yb)

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    # The first step doubles as the compile probe (graph build; with --compile, compilation).
    # Its loss value is unaffected by the timing, so it is also parity step 0.
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

    # ROCm builds alias HIP onto torch's "cuda" device; label the report row honestly.
    backend = "cuda(hip)" if args.device == "cuda" and torch.version.hip else args.device
    result = {
        "framework": "pytorch",
        "backend": backend,
        "variant": "compiled" if args.compile else "eager",
        "workload": meta["name"],
        "compile_s": round(compile_s, 3),
        # Whether inductor's codegen ran here or came from its cache (gh-ocannl-644). A compiled
        # cell compiles in the process that then times steps, unlike an OCANNL tuned cell; the
        # report states that rather than leaving it assumed (gh-ocannl-675).
        "searched": torch_searched(torch, args.compile),
        "step_ms": percentiles(synced),
        "queued_step_ms": queued,
        "timed_steps": timed_steps,
        "losses": losses,
        "version": torch.__version__,
        # What this process ran under, for the sweep's regime gate and the report (gh-ocannl-719):
        # derived from torch's effective settings above, not echoed from --regime. `runner_regime`,
        # not `regime`: the sweep stamps `regime` with what it dispatched, and a same-named key
        # would let the stamp mask a runner that ran the other arm.
        "runner_regime": runner_regime,
        "regime_settings": regime_settings,
    }
    if args.compile_mode:
        result["compile_mode"] = args.compile_mode
    if retimed:
        result["retime_step_ms"] = percentiles(retimed)
    if tokens_per_step:
        result["tokens_per_step"] = tokens_per_step
    emit(result)


if __name__ == "__main__":
    main()
