# Study krnl and autograph

Issue: [#277](https://github.com/ahrefs/ocannl/issues/277)

## Goal

Write a short comparison of krnl/autograph, Luminal, and current OCANNL,
focused on lessons that remain actionable.

## Updated framing

The original proposal was written before OCANNL gained parallel schedule
transforms, autotuning, pooled/liveness-planned memory, HIP, and tensor
persistence. Serialization, “single-threaded kernels,” and the missing pool
allocator are no longer open design inputs. A generic Vulkan fallback is also
not justified merely because krnl used one.

The useful questions now are:

- What made krnl's Rust-to-SPIR-V kernel authoring pleasant or limiting?
- Where did autograph's eager tape and derive-based model composition help,
  and where did the project stop scaling?
- How does Luminal's graph rewriting compare with OCANNL's lower-level
  schedule optops and seeded autotuning?
- Which maintenance, portability, and project-scope choices explain why the
  projects stalled or evolved?

## Deliverable

A durable research note that:

- identifies the exact revisions studied;
- compares the three systems at the kernel, graph/autodiff, memory, model
  composition, and deployment layers;
- separates already-solved differences from genuine gaps;
- recommends only findings backed by a concrete OCANNL use case or measured
  pain point;
- records explicit “do not pursue” conclusions where appropriate.

Do not require one follow-up issue per observation or predict the conclusions
in advance. File follow-ups only for ideas that survive the comparison and
have a plausible owner and validation strategy. The study is complete when
the note is committed and summarized on #277.
