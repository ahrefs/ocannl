# Add LSTM example (gh-ocannl-60)

## Status update (2026-06-12)

- Issue [#60](https://github.com/ahrefs/ocannl/issues/60) is **OPEN**, milestone v0.7.1 (past due); harness task is `deferred`. No LSTM code exists anywhere in the repo (`lib/nn_blocks.ml`, `test/training/` re-verified) — the proposal is entirely unimplemented.
- Related issues: gh-ocannl-57 (transformer on Names) **CLOSED/merged** as the proposal already noted; gh-ocannl-59 (makemore examples) now **CLOSED COMPLETED** — `mlp_names.ml`, `mlp_bn_names.ml`, and `batch_norm1d` landed, so `test/training/` has more scaffold precedents than when this was written.
- **Key new capability**: a tensor stacking operation + block-literal `%op` syntax landed (commit `58bfd6e5`); `Operation.stack` is at `tensor/operation.ml:508` and exposed through the DSL surfaces. This supersedes the hand-rolled `stack_along_seq` sketch in Phase 4 — use `stack` to reassemble the per-step hidden states. (Note: print `stack`/`concat` results with `~style:\`Default`, not `\`Inline` — the inline printer crashes on concatenated shapes.)
- Line drift in `tensor/operation.ml`: `slice` is now defined at line 617 (was 584), `@|` exposed in the `O` modules around line 858 (was 819).
- `transformer_names.ml` scaffold remains valid; its cross-entropy was switched to a numerically stable log-softmax (commit `4daab9ee`), matching the proposal's "max-shift log-softmax" description.
- No `sigmoid` primitive has been added since; the option-1-vs-option-2 design question is still open.
- Repo-wide renames since April 2026 (broadcast-order reversal LUB→GLB, "label"→"basis") do not affect this plan; write any new shape-inference text with post-reversal vocabulary ("refines", join semilattice).

## Goal

Add OCANNL's first recurrent architecture: an LSTM building block and a
training/generation example that mirrors the existing transformer-on-Names
setup. Demonstrate that OCANNL's static compilation model supports recurrent
networks via unrolled graphs with parameter-shared cells.

Issue: https://github.com/ahrefs/ocannl/issues/60

> "Maybe use the language modelling setup from the Transformer example." — @lukstafi

## Acceptance Criteria

- [ ] An `lstm_cell` building block is added to `lib/nn_blocks.ml`, exposing
      the standard LSTM equations (forget / input / output gates, candidate
      cell, cell-state update, hidden output). It returns the new `(h, c)`
      pair from a previous `(h_prev, c_prev)` and an input `x`.
- [ ] An `lstm` higher-level block unrolls `lstm_cell` over a fixed
      `seq_len`, sharing parameters across all time steps through OCANNL's
      unit-parameter lifting.
- [ ] Weight sharing is verified: the per-time-step forward graph references
      a single set of parameter tensors for `W_*` / `b_*` (not `seq_len`
      independent copies).
- [ ] `test/training/lstm_names.ml` trains a character-level language model
      on the Names dataset, following the data-prep and language-modeling
      setup established in `test/training/transformer_names.ml` (same
      `ctx_len`, `pad_char`, `bos_idx`, one-hot encoding, cross-entropy on
      softmax logits).
- [ ] After training, the example generates names autoregressively by
      running the cell one step at a time, threading hidden/cell state
      between steps. Generated samples are recognizably name-like (same
      qualitative bar as `bigram.ml` / `transformer_names.ml`).
- [ ] Example builds and runs on the CPU backend (parity with other
      `test/training/*_names.ml` examples). Expected output is checkpointed
      in `lstm_names.expected`.
- [ ] Documentation: if `sigmoid` is introduced as a new primitive in
      `tensor/operation.ml`, it is exported symmetrically with `tanh` /
      `sat01` (both `TDSL` and `NTDSL` surfaces).

## Context

### Verified against current HEAD (commit `02ecacff`, `master`)

**Existing infrastructure (reuse):**

- **Language modeling setup** — `test/training/transformer_names.ml` is
  merged (gh-ocannl-57). It provides: `ctx_len`/`eff_seq_len`/`vocab_size`
  constants, `name_to_sequences`, `seqs_to_flat_one_hot`, input/target batch
  tensor shapes `batch_dims:[batch_size; eff_seq_len] output_dims:[vocab_size]`,
  cross-entropy loss (max-shift log-softmax), SGD wiring, and the
  per-position inference loop (re-runs a fixed-length routine, reads the
  logit at the current position, samples, writes back). The LSTM example can
  copy this scaffold almost verbatim.
- **Dataset** — `Dataprep.Names` (`dict_size = 28`, `char_index`,
  `letters_with_dot`, `read_names`).
- **Parameter lifting** (`docs/syntax_extensions.md` §"Need to lift the
  applications of configuration arguments") — inline `{ w }` / `{ b = …; o = … }`
  declarations inside a `let%op cell ~label ~hidden_dim () = fun … → …` are
  created exactly once at the `()` application. Applying `cell` at every
  time step therefore shares the same tensor nodes. This is the mechanism
  that makes unrolled LSTMs correct without any new framework support.
- **Batch-axis slicing** — `operation.ml` `slice` / infix `@|` (defined at
  line 617, exposed in the `O` modules around line 858) *(Update 2026-06-12:
  line numbers refreshed)* takes an
  `Indexing.static_symbol` and returns the tensor sliced at the leftmost
  batch axis. `@|` is the idiomatic way to extract time step `t` from a
  `[batch_size; seq_len]` batch layout. The einsum notation `"2...|... => ...|..."`
  (syntax_extensions.md line 461) provides a literal-index alternative.
- **Unary op primitives** — `tanh` exists in `tensor/operation.ml`
  (`op_label:"tanh"`). `sat01` is saturation to [0, 1], not sigmoid. **No
  `sigmoid` primitive exists** (verified by grep across `lib/`, `tensor/`,
  `arrayjit/lib/`).
- **Building-block conventions** — `nn_blocks.ml` exposes layers as
  `let%op name ~label ~cfg () = fun args → …`. `mlp_layer` (line 78) is the
  minimal reference; `multi_head_attention` (line 181) shows a multi-parameter
  cell with gated pos-embedding options.
- **Inference scaffolding** — `test/training/bigram.ml` `infer` closure
  (around line 85) shows how to (a) write a new one-hot input into an input
  tensor via `Tn.set_values`, (b) run a compiled routine, (c) read the
  output probability array, (d) sample via a dice tensor. `transformer_names.ml`
  `aux` loop (around line 226) shows the same pattern for a fixed-length
  sequence with position-based logit reads.

### Design question surfaced by verification

**No `sigmoid` primitive.** Two paths:

1. Define locally in `nn_blocks.ml`: `let%op sigmoid x = recip (1 + exp (neg x))`.
   Adds ~4 ops per gate × 4 gates per step × `seq_len` → ~256 extra nodes at
   `seq_len=16`. Fine at this scale.
2. Add `sigmoid` as a proper primitive in `tensor/operation.ml` (mirroring
   `tanh`: custom `op_asn` / `grad_asn` that use the identity
   `sigmoid'(x) = sigmoid(x) * (1 − sigmoid(x))` for numerical stability and
   single-pass backward).

Option 2 is ~20 lines and benefits any future RNN/attention variant; option 1
keeps the scope tight. **Open for user preference.** Default to option 1 to
keep this PR focused; file a follow-up for option 2 if desired.

### Related tasks

- **Blocks/Relates**: gh-ocannl-57 (decoder transformer on Names) — **merged
  at HEAD**, so the `blocked_by` linkage in the task frontmatter is stale and
  should be cleared after this proposal lands.
- **gh-ocannl-49** (concat `^` syntax) — *no longer a hard dependency*; this
  proposal uses the separate-weight-matrices formulation (Approach B), which
  sidesteps concatenation entirely.
- **gh-ocannl-59** (makemore examples) — LSTM is the RNN variant in the
  makemore lineage. *(Update 2026-06-12: #59 is closed as completed;
  `mlp_names.ml` and `mlp_bn_names.ml` landed in `test/training/`.)*

## Approach

*Suggested approach — agents may deviate if they find a better path.*

### Approach B: separate `W_x`, `W_h` matrices (no concatenation)

Each gate computes `W_x · x + W_h · h_prev + b` rather than the textbook
`W · [x; h_prev] + b`. Mathematically equivalent, and avoids any dependency
on concat syntax. OCANNL's einsum handles the two matmuls naturally, and the
per-gate parameter count is the same as the concatenated form.

```ocaml
(* lib/nn_blocks.ml — sketch, not final syntax *)
let%op sigmoid x = recip (1 + exp (neg x))

let%op lstm_cell ~label ~hidden_dim () =
  fun x ~h_prev ~c_prev ->
    let f = sigmoid ({ w_xf } * x + { w_hf } * h_prev + { b_f = 0.; o = [hidden_dim] }) in
    let i = sigmoid ({ w_xi } * x + { w_hi } * h_prev + { b_i = 0.; o = [hidden_dim] }) in
    let o = sigmoid ({ w_xo } * x + { w_ho } * h_prev + { b_o = 0.; o = [hidden_dim] }) in
    let g = tanh    ({ w_xg } * x + { w_hg } * h_prev + { b_g = 0.; o = [hidden_dim] }) in
    let c = (f *. c_prev) + (i *. g) in
    let h = o *. tanh c in
    (h, c)
```

Parameter lifting places all `w_*` / `b_*` at the `()` call, so one call to
`lstm_cell ~hidden_dim ()` produces a closure with eight weight tensors plus
four biases, reused across every time step.

### Unrolling (Phase 2)

```ocaml
let lstm ~label ~hidden_dim ~seq_len () =
  let cell = lstm_cell ~label:("cell" :: label) ~hidden_dim () in
  fun input_seq ~h0 ~c0 ->
    let rec step t h c outs =
      if t >= seq_len then List.rev outs
      else
        let x_t = input_seq @| (time_step_var t) in
        let h', c' = cell x_t ~h_prev:h ~c_prev:c in
        step (t + 1) h' c' (h' :: outs)
    in
    step 0 h0 c0 []
```

The `time_step_var` uses a `Indexing.static_symbol` per step (or a literal
numeric index in einsum — see Phase 3). Returns the list of hidden states,
which is then stacked or used directly depending on the caller.

### Time-step extraction (Phase 3)

Two viable mechanisms:

1. **`@|` slice at a static symbol** — one static symbol per time step,
   bound once in the routine's bindings, driving the `seq_len` slice
   instances. Simpler conceptually, but creates `seq_len` symbols.
2. **Literal-index einsum** — `"t_literal...|... => ...|..."` generates a
   fresh unary op per time step with the dimension baked in. Equivalent at
   the graph level.

Start with (1) because it's already used elsewhere in the codebase for batch
slicing. If the symbol overhead bites, fall back to (2).

### Training example (Phase 4)

Copy `test/training/transformer_names.ml` wholesale as the scaffolding:

- Same `ctx_len = 16`, `vocab_size`, `pad_char`, `bos_idx`.
- Same `name_to_sequences` / `seqs_to_flat_one_hot` / `make_data_tensor`.
- Same max-shift log-softmax cross-entropy (compute loss at every position).
- Replace the `let open Nn_blocks; let mha = …; let ffn = …; let%op build_model = …`
  block with:

  ```ocaml
  let lstm_block = Nn_blocks.lstm ~label:["lstm"] ~hidden_dim ~seq_len:eff_seq_len () in
  let%op build_model () =
    fun input ->
      let embedded = { tok_embed; o = [ hidden_dim ] } * input in
      let h0 = (* zero tensor, batch_dims=[batch_size], output_dims=[hidden_dim] *) in
      let c0 = (* likewise *) in
      let hs = lstm_block embedded ~h0 ~c0 in
      (* stack hs back along a seq_len batch axis, then project *)
      { w_out } * (stack_along_seq hs)
  ```

  `stack_along_seq` is the inverse of the per-step slicing — can be expressed
  via einsum `concat_sum`, or by writing the hidden-state outputs directly
  into a pre-allocated tensor using named axes. Coder decides.
  *(Update 2026-06-12: a first-class `stack` operation has since landed
  (`tensor/operation.ml:508`, commit `58bfd6e5`) — prefer it over the
  hand-rolled alternatives above.)*

- SGD, epochs, step counter, expected-file generation all unchanged.

### Autoregressive generation (Phase 5)

Build a second compiled routine for single-step inference:

```ocaml
let infer_h = (* input tensor, batch=[1], output=[hidden_dim] *) in
let infer_c = (* likewise *) in
let infer_x = (* input one-hot, batch=[1], output=[vocab_size] *) in
let cell = (* same closure used during training — parameter sharing persists *) in
let h', c' = cell (embed infer_x) ~h_prev:infer_h ~c_prev:infer_c in
let logits = { w_out } * h' in
```

Between generation steps, copy `h'`/`c'` back into `infer_h`/`infer_c` using
`Tn.set_values` (`host_to_device` round-trip), following the `bigram.ml`
per-step state update pattern. Sample from logits with the same dice-tensor
approach used in `transformer_names.ml`.

### Initial hidden/cell state

Zero tensors via `NTDSL.init ~l:"h0" ~prec:Ir.Ops.single ~b:[batch_size] ~o:[hidden_dim] ~f:(fun _ → 0.) ()`
(same pattern the transformer uses for the causal mask).

### Deliberately out of scope

- Gradient clipping (LSTMs often benefit but OCANNL's SGD doesn't currently
  support it; add only if training diverges).
- Variable sequence lengths / dynamic unrolling.
- Padding masks for the loss (pad tokens contribute to loss — acceptable for
  short names at `seq_len=16`, matching what transformer_names.ml already
  does).
- Multi-layer / bidirectional LSTM (single-layer, left-to-right only).
- GRU, other RNN variants.

## Scope

**In scope**

- `lib/nn_blocks.ml`: `lstm_cell`, `lstm`, optionally local `sigmoid`
  (default) OR primitive `sigmoid` in `tensor/operation.ml` (if user
  prefers).
- `test/training/lstm_names.ml` + `lstm_names.expected` + `dune` entry.
- Ocamldoc comments on the new `nn_blocks.ml` exports.

**Out of scope** (file follow-ups if encountered)

- Any new indexing / slicing primitives — existing `@|` + einsum is
  sufficient.
- Concatenation axis work (gh-ocannl-49) — sidestepped by Approach B.
- Multi-layer LSTM, GRU, bidirectional variants — future work.
- RNN-specific optimization passes (recurrence detection exists in IR but is
  used only for correctness/visit tracking; no action required here).

**Dependencies**

- gh-ocannl-57: **already merged at HEAD**; can be removed from
  `blocked_by`.
- gh-ocannl-49: no longer required (Approach B bypasses it).
