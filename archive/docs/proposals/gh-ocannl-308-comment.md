# Sasha Rush's Tensor Puzzles in OCANNL

Worked through all 21 [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) in OCANNL, preferring
extended einsum notation. The solutions are committed as an executable, self-checking test at
`test/einsum/tensor_puzzles.ml` (golden output in `test/einsum/tensor_puzzles.expected`), so they
stay correct as OCANNL evolves.

**Result: 8 solvable in einsum / `%op`, 10 with op-level workarounds (18 solvable total), 3 not expressible today.**

**Key principle** (thanks to review feedback): any *static, data-independent* index map — a
permutation, reshape, prefix sum, or convolution — is a dense selector/operator matrix `S[i,o]`
built from index-grid comparisons and contracted via einsum (`a +* "i; i o => o" S`). It is
`O(size²)`, not how you'd implement it for real, but it *is* expressible. Only *data-dependent*
index maps (where the index is itself a runtime tensor value) genuinely need gather/scatter, which
OCANNL lacks. So cumsum, flip, roll, and flatten move to the solvable tier; only compress, bincount,
and scatter_add remain out of reach.

## A. Solvable with OCANNL einsum / `%op`

1. `ones` `(i)->[i]` — broadcast a constant: `(range i *. 0.) + 1`
2. `sum` `([i])->[1]` — unary reduction: `a ++ "i => 0"`
3. `outer` `([i],[j])->[i,j]` — no shared index ⇒ outer product: `a +* "i; j => i->j" b`
4. `diag` `([i,i])->[i]` — repeated-label extraction: `m ++ "ii => i"` (OCANNL's `einsum1` extracts diagonals — the proposal's old "not expressible" was wrong)
9. `vstack` `([i],[i])->[2,i]` — block-literal stacking: `[ a; b ]`
19. `heaviside` — nested pointwise `where (a = 0) b (where (0 < a) 1 0)`
20. `repeat` `([i])->[d,i]` — block-literal stacking of `d` copies: `[ a; a; a ]`
21. `bucketize` `([i],[j])->[i]` — broadcast compare + reduce: `not (v < bnd) ++ "ij => i"` (counting boundaries ≤ v; `>=` is expressed as `not (<)`)

## B. Solvable with op-level workarounds (index-grid selector matrices, concatenation) — static sizes

5. `eye` `(j)->[j,j]` — row grid `[n,1]` vs column grid `[1,n]`, equality: `(r ++ "i=>i0") = (r ++ "j=>0j")`
6. `triu` `(j)->[j,j]` — same grids, `i <= j` as `not ((r ++ "j=>0j") < (r ++ "i=>i0"))`
7. `cumsum` `([i])->[i]` — lower-triangular ones selector `L[i,o]=(i<=o)`, contracted: `a +* "i; i o => o" L`
8. `diff` `([i])->[i-1]` — finite-difference operator matrix `D[i,o] = (i = o+1) - (i = o)` built from index grids, then contracted: `a +* "i; i o => o" D`. (The natural valid-convolution spelling `a +* "o<+k; k => o" kernel` currently **hangs shape inference** even on a length-5 input — see below.)
10. `roll` `([i])->[i]` — circular permutation selector `S[i,o]=(i=o+1)` plus the wrap term `(i=0)·(o=n-1)`, contracted
11. `flip` `([i])->[i]` — anti-diagonal selector `S[i,o]=(i+o=n-1)`, contracted
13. `pad_to` `([i],j)->[j]` — both directions for static target sizes: pad via concatenation `(a, zeros) ++^ "i; j => i^j"`, truncate via a rectangular selection matmul `a +* "i; i o => o" sel`, `sel[i,o]=(i=o)`. Only a *runtime* (data-dependent) target `j` remains a gap.
14. `sequence_mask` `([i,j],[i])->[i,j]` — column index grid vs per-row length: `where ((r ++ "j=>0j") < (len ++ "i=>i0")) values 0`
17. `flatten` `([i,j])->[i*j]` — static reshape via selector `S[i,j,k]=(i*cols+j = k)` (using `range_of_shape` for the row-major offset `i*cols+j`), contract `i,j`: `a +* "i j; i j k => k" sel`
18. `linspace` `(lo,hi,n)->[n]` — affine arithmetic on a range: `(range n *. step) + lo`, `step = (hi-lo)/(n-1)` (the `n=1` case divides by zero and is handled separately)

## C. Not expressible today — data-dependent indexing

The output index is itself a runtime tensor value, so a static selector cannot encode it — this needs
gather/scatter, which OCANNL lacks:

12. `compress` — left-align kept entries: the output position of each kept element is the prefix-count of the mask (a runtime value), then **scatter**. (Note: cumsum *is* now expressible, but the subsequent scatter to data-dependent positions is not.)
15. `bincount` — `out[a[i]] += 1`; **scatter** with values-as-indices (and integer index tensors)
16. `scatter_add` — `out[link[j]] += values[j]`; the canonical **scatter**

## Most impactful missing capabilities

1. **Gather / scatter (dynamic indexing)** — the *only* fundamental gap; unlocks compress (12), bincount (15), scatter_add (16); needed for embedding lookup, variable-length masking, and many real ML ops. Related: #293 (sharding & slicing).
2. **Scan / prefix-sum** — not needed for *expressibility* (cumsum is the triangular-selector matmul above), but it is the `O(n)` streaming way; the selector matmul is `O(n²)`.
3. **Native reshape/view, reverse, and modular/affine indexing** — ergonomic, efficient replacements for the `O(size²)` selector-matrix workarounds used for flatten/flip/roll.

## Bonus finding (shape-inference bug)

The 1-D valid-convolution einsum `a +* "o<+k; k => o" kernel` over output-only axes **hangs**
OCANNL's shape inference (no error, just spins) even on a length-5 vector with a length-2 kernel.
The operator-matrix formulation of `diff` works around it; worth filing separately.
