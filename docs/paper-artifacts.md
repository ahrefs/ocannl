# Paper artifacts

**No conference submission is scheduled.** The OCaml Workshop / FProPer submission was not accepted, and IFL 2026 was considered and decided against as a poor fit. The written material below is therefore maintained for its own sake — as the project's technical account, and as the starting point should a venue be chosen later. Only the formal core technical report and the constraint-generation notes are live work; the rest is archival.

The paper-facing material, first assembled for the v0.7/v0.8 workshop submission:

- Workshop article: [docs/ocannl_workshop_article_human.md](ocannl_workshop_article_human.md) — **historical artifact**, kept as-is; it describes the project as of the [0.8 release](https://github.com/ahrefs/ocannl/releases/tag/0.8) and was written for the OCaml Workshop / FProPer submission, which was not accepted.
- Workshop article PDF: [docs/html/pdfs/ocannl_workshop_article_human.pdf](html/pdfs/ocannl_workshop_article_human.pdf) — likewise archival, rendered from the v0.8-era source.
- Formal core technical report: [rendered PDF](https://ahrefs.github.io/ocannl/docs/pdfs/ocannl-formal-core-technical-report.pdf), LaTeX source in [docs/](ocannl-formal-core-technical-report.latex) — live, still developing.
- Shape constraint generation notes: [docs/shape-constraint-generation.md](shape-constraint-generation.md) — live.

The intended shape of a paper, should one be written, is recorded below.

### Proposed Title
*"Generalized Einsum with Row Variables: Shape Inference for Deep Learning in OCaml"*

### Key Contributions
1. **Generalized einsum notation** with convolutions, strided iteration, and concatenation
2. **Row variables** for flexible axis handling ("principle of least commitment")
3. **Constraint-based shape inference** with provenance tracking for error messages
4. **Dimension basis** design rationale (vs. axis labels)
5. **Integration with OCaml's type system** via syntax extensions

### Related Work to Address
- einops (#413)
- torchdim / DumPy (#316)
- Named tensors in PyTorch/JAX
- Dependent types for tensor shapes

### Why v0.7 Was the Prerequisite
Any such paper needs working examples on OCANNL's mature frontend, all delivered by v0.7:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

The deep semantic groundwork (the two-sorted ground algebra, the rank-fact graph and rank-cycle check, ≈-semantics for row equality) now lives in the formal core technical report and its appendix.
