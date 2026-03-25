## Deep dive into DumPy and torchdim — findings

I've done a detailed study of [DumPy](https://dynomight.net/dumpy/) and [torchdim](https://github.com/facebookresearch/torchdim) (now `functorch.dim` in PyTorch). Full write-up in `docs/proposals/gh-ocannl-316.md`. Here's the summary.

### Three-way comparison

| Aspect | OCANNL | DumPy | Torchdim |
|--------|--------|-------|----------|
| Axis identification | Kind (B/I/O) + position | External string labels | First-class Dim objects |
| Label scope | Local to einsum spec | Temporary (per expression) | Object identity (variable scope) |
| Broadcasting | Implicit (kind matching + row vars) | Explicit only (no implicit) | Implicit (union of dims) |
| Einsum | String spec: `"i,j;j,k=>i,k"` | `@` + named indices | Multiply + sum (pattern-matched) |
| Index arithmetic | Affine indices in einsum | Loop variables as values | Dims as tensors |
| Unknown axes | Row variables (`...`) | Not supported | Not supported |

### Design conclusion

Positional axes with optional dimension units are the right choice for OCANNL specifically because OCANNL is built around einsum notation with constraint-based shape inference and row variables:

1. **Einsum is inherently positional** — mirrors mathematical tradition directly
2. **Row variables need positional structure** — "some unknown axes with unknown names" can't work with named-axis systems
3. **Inference ambiguity** — when two axes share the same semantic unit (two spatial dims both meaning "position"), named systems must disambiguate; positional systems don't
4. **Conciseness** — `"i,j;j,k=>i,k"` vs `(A[i,k]*B[k,j]).sum(k).order(i,j)`

Named dimensions have genuine advantages for imperative tensor programming (self-documenting code, transposition safety, elegant implicit batching). The argument is specifically about einsum-based declarative systems, not a blanket claim.

### Follow-up ideas (prioritized)

1. **Dims-as-tensors / arange primitive** (high priority) — Torchdim's Rule 3 lets a dimension act as `[0, 1, ..., n-1]`, enabling `eye`, `triu`, `diag`, and sequence masks. OCANNL currently can't express these. This directly addresses the gap identified in #308.

2. **Strict broadcasting mode** (medium) — DumPy forbids implicit broadcasting entirely. OCANNL could offer an opt-in mode requiring all broadcasting to be explicit in einsum specs, catching silent shape bugs.

3. **Loop mental model in docs** (medium) — Both DumPy and torchdim use "think of it as a loop" as their teaching tool. Our docs should emphasize loop-equivalent readings of einsum specs.

### Workshop paper impact

The related work section of the workshop paper (#299) has been updated to distinguish three named-dimension approaches (string-named, external-label, object-identity) instead of the previous single-paragraph summary. The dimension units section now includes two additional arguments (row variable compatibility, conciseness).

Full proposal: `docs/proposals/gh-ocannl-316.md`
