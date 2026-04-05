# Sasha Rush Tensor Puzzles in OCANNL

**Issue:** [ahrefs/ocannl#308](https://github.com/ahrefs/ocannl/issues/308)
**Status:** Proposal

## Goal

Solve the [Sasha Rush Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) that map naturally to OCANNL's extended einsum notation, and provide clear explanations for those that do not fit. The deliverable is a documented analysis (posted as a GitHub issue comment and committed as a test/example file) that showcases OCANNL's einsum expressiveness and identifies gaps.

## Acceptance Criteria

- [ ] All 21 puzzles categorized into three tiers: (1) solvable with OCANNL einsum/`%op`, (2) solvable with OCANNL ops but requiring workarounds, (3) not expressible
- [ ] Working OCANNL code for every solvable puzzle, preferring einsum notation where possible, committed as a test file `test/einsum/tensor_puzzles.ml` with inline expect tests verifying correctness on small inputs
- [ ] For each non-expressible puzzle, a concise explanation of which OCANNL capability is missing (dynamic indexing, scan/prefix-sum, reshape, etc.)
- [ ] A summary table identifying which missing capabilities would be most impactful to add, with links to existing issues where applicable
- [ ] The full analysis posted as a comment on GitHub issue #308

## Context

### What the puzzles are

Sasha Rush's Tensor Puzzles are 21 exercises that reimplement standard NumPy/PyTorch functions using only broadcasting, `arange(i)`, `where(q, a, b)`, arithmetic/comparison operators, and `@` (matmul). They teach tensor programming from first principles through broadcasting tricks.

### OCANNL's relevant capabilities

OCANNL's extended einsum provides:
- Three axis kinds: `batch | input -> output`
- Row variables (`...`) for broadcasting
- Axis concatenation (`^`) for stacking and slicing
- Affine indexing (`stride*out+kernel`, `stride*out<+kernel`) for convolutions
- Pointwise ops: arithmetic, comparison (`<`, `=`, `<>`), logic (`&&`, `||`, `not`), `where`
- Reduction operators: `++` / `+++` (sum), `@^^` (max), `++^` (concatenation)
- `range_of_shape` for creating tensors filled with index values

### Classification of puzzles

| # | Puzzle | OCANNL Status | Key mechanism |
|---|--------|---------------|---------------|
| 1 | `ones` | **Einsum** | Constant tensor |
| 2 | `sum` | **Einsum** | `+++ " => 0"` reduction |
| 3 | `outer` | **Einsum** | `+* "i;j => ij"` outer product |
| 4 | `diag` | Not expressible | Needs diagonal extraction (repeated index = trace, not diag) |
| 5 | `eye` | **Workaround** | `range_of_shape` comparison: `where(i = j, 1, 0)` |
| 6 | `triu` | **Workaround** | `range_of_shape` comparison: `where(i <= j, 1, 0)` |
| 7 | `cumsum` | Not expressible | Loop-carried dependency, needs scan primitive |
| 8 | `diff` | **Workaround** | Convolution with kernel `[1, -1]` via affine indexing |
| 9 | `vstack` | **Einsum** | Concatenation: `++^ "a; b => a^b"` |
| 10 | `roll` | Not expressible | Circular shift needs modular dynamic indexing |
| 11 | `flip` | Not expressible | Reversed indexing `a[n-1-i]` not in affine spec |
| 12 | `compress` | Not expressible | Prefix-sum for output positions + scatter |
| 13 | `pad_to` | **Workaround** | Concatenation `^` for padding, axis slicing for truncation |
| 14 | `sequence_mask` | **Workaround** | `where(j_index < length[i], values[i][j], 0)` via broadcasting |
| 15 | `bincount` | Not expressible | Scatter-add with tensor values as indices |
| 16 | `scatter_add` | Not expressible | Dynamic indexing: `out[link[j]] += values[j]` |
| 17 | `flatten` | Not expressible | Reshape between differently-shaped tensors |
| 18 | `linspace` | **Workaround** | Arithmetic on `range_of_shape` index values |
| 19 | `heaviside` | **Einsum** | Pointwise `where(a = 0, b, where(a > 0, 1, 0))` |
| 20 | `repeat` | **Einsum** | Broadcasting: batch axis expansion |
| 21 | `bucketize` | **Einsum** | Broadcast compare + sum: `(v >= boundaries) +++ "i,j => i"` |

**Summary:** 7 solvable with einsum, 6 with workarounds (13 total), 8 not expressible.

### Missing capabilities identified

| Missing Capability | Unlocks Puzzles | Impact |
|--------------------|-----------------|--------|
| **Gather/scatter (dynamic indexing)** | #4 diag, #15 bincount, #16 scatter_add, #12 compress | High -- needed for embedding lookup, attention, many real ML ops |
| **Scan/prefix-sum** | #7 cumsum, #12 compress | Medium -- sequence processing, CTC loss |
| **Reverse/flip indexing** | #11 flip | Low -- negative-stride affine extension |
| **Circular shift** | #10 roll | Low -- modular affine index |
| **Reshape/flatten** | #17 flatten | Medium -- tensor reshaping between views |

### Approach

1. Create `test/einsum/tensor_puzzles.ml` with inline expect tests, one `%expect_test` per solvable puzzle
2. Each puzzle solution uses `%op` syntax with einsum notation where possible, falling back to `TDSL`/`NTDSL` operations where needed
3. Use `range_of_shape` to create index-value tensors for workaround puzzles (eye, triu, sequence_mask, linspace)
4. Use `++^` concatenation for vstack and pad_to
5. Use affine convolution indexing for diff (kernel `[1, -1]`)
6. Include comments explaining the non-expressible puzzles inline
7. Post the full classification table and code as a comment on issue #308

### Key design decisions

- **Test file, not documentation**: Solutions live as executable tests, ensuring they stay correct as OCANNL evolves
- **Float domain**: All puzzles are solved in OCANNL's float domain; integer-semantic puzzles (bincount, scatter_add) are noted as requiring float-to-int casting awareness
- **Boundary conditions**: Puzzle #8 (diff) first element handling -- the convolution naturally handles boundaries via valid-convolution semantics (output shorter than input) or padding
- **No new primitives**: This task is purely about demonstrating what OCANNL can already do and documenting the gaps; no new operations are added
