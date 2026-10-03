# Research notes

Deep dives, feasibility studies, and lessons memos — documents whose content *is*
the research, as opposed to `docs/proposals/` which holds plans for changes (including
plans to *do* research whose eventual write-up lands here).

Two kinds of entries:

- **Regular files**: notes authored directly in this directory
  (e.g. `superoptimizers.md`, the #261 evidence base).
- **Symlinks**: research notes that live elsewhere and are indexed here. Notes under
  `docs/proposals/` stay there because their task-id basenames map to harness tasks
  and they carry dense same-directory relative links; two older deep dives live at
  the `docs/` root and are indexed here without moving them.

| Entry | Points to | Topic |
|---|---|---|
| `tinygrad-deep-dive.md` | `../proposals/tinygrad-deep-dive.md` | tinygrad architecture comparison; article at `docs/blog/a-range-is-not-its-shape.md` |
| `distro-feasibility-study.md` | `../proposals/distro-feasibility-study.md` | DisTrO/DeMo distributed training feasibility (#278) |
| `lean-attention-feasibility.md` | `../proposals/gh-ocannl-263.md` | Lean Attention / Flash Attention as softmax-reduce (#263) |
| `dumpy-torchdim-deep-dive.md` | `../proposals/gh-ocannl-316.md` | DumPy & torchdim dimension-naming comparison (#316; findings comment in `gh-ocannl-316-comment.md`) |
| `imbue-infrastructure-lessons.md` | `../imbue-infrastructure-lessons.md` | Imbue 70B infrastructure lessons (#270) |
| `megakernel-deep-dive.md` | `../megakernel-deep-dive.md` | Megakernel patterns (Hazy Research, Mirage MPK) (#318) |
| `ggml-lessons.md` | (regular file) | Efficiency lessons from ggml for CPU inference (#163) |
| `llmc-lessons.md` | (regular file) | llm.c lessons for GPU training/inference of the GPT-2 driver workload (#253) |
| `gh-1099-register-tile-c-traffic.md` | (regular file) | Register-tile C-traffic invariant, AVX2 A/B and the remaining tail-bearing tie question (gh-ocannl-1099, gh-ocannl-1180) |
