# Sasha Rush's Tensor Puzzles in OCANNL

Worked through all 21 [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) in OCANNL, preferring
extended einsum notation. The solutions are committed as an executable, self-checking test at
`test/einsum/tensor_puzzles.ml` (golden output in `test/einsum/tensor_puzzles.expected`), so they
stay correct as OCANNL evolves.

**Result: 8 solvable in einsum / `%op`, 6 with op-level workarounds (14 solvable total), 7 not expressible today.**

## A. Solvable with OCANNL einsum / `%op`

1. `ones` `(i)->[i]` — broadcast a constant: `(range i *. 0.) + 1`
2. `sum` `([i])->[1]` — unary reduction: `a ++ "i => 0"`
3. `outer` `([i],[j])->[i,j]` — no shared index ⇒ outer product: `a +* "i; j => i->j" b`
4. `diag` `([i,i])->[i]` — repeated-label extraction: `m ++ "ii => i"` (OCANNL's `einsum1` extracts diagonals — the proposal's old "not expressible" was wrong)
9. `vstack` `([i],[i])->[2,i]` — block-literal stacking: `[ a; b ]`
19. `heaviside` — nested pointwise `where (a = 0) b (where (0 < a) 1 0)`
20. `repeat` `([i])->[d,i]` — block-literal stacking of `d` copies: `[ a; a; a ]`
21. `bucketize` `([i],[j])->[i]` — broadcast compare + reduce: `not (v < bnd) ++ "ij => i"` (counting boundaries ≤ v; `>=` is expressed as `not (<)`)

## B. Solvable with op-level workarounds (index grids / convolution / concatenation)

5. `eye` `(j)->[j,j]` — row grid `[n,1]` vs column grid `[1,n]`, equality: `(r ++ "i=>i0") = (r ++ "j=>0j")`
6. `triu` `(j)->[j,j]` — same grids, `i <= j` as `not ((r ++ "j=>0j") < (r ++ "i=>i0"))`
8. `diff` `([i])->[i-1]` — finite-difference operator matrix `D[i,o] = (i = o+1) - (i = o)` built from index grids, then contracted: `a +* "i; i o => o" D`. (The natural valid-convolution spelling `a +* "o<+k; k => o" kernel` currently **hangs shape inference** even on a length-5 input — see gaps below.)
13. `pad_to` `([i],j)->[j]` — both directions for statically-known target sizes: pad via concatenation `(a, zeros) ++^ "i; j => i^j"`, truncate via a rectangular selection matmul `a +* "i; i o => o" sel` with `sel[i,o]=(i=o)`. Only a *runtime* (data-dependent) target `j` remains a gap — that needs a dynamic/range-slice op.
14. `sequence_mask` `([i,j],[i])->[i,j]` — column index grid vs per-row length: `where ((r ++ "j=>0j") < (len ++ "i=>i0")) values 0`
18. `linspace` `(lo,hi,n)->[n]` — affine arithmetic on a range: `(range n *. step) + lo`, `step = (hi-lo)/(n-1)` (the `n=1` case divides by zero and is handled separately)

## C. Not expressible today

Each names the missing primitive:

7. `cumsum` — loop-carried prefix dependency; needs a **scan** primitive
10. `roll` — `a[(i+1) mod n]`; needs **gather** / circular shift (modular dynamic index)
11. `flip` — `a[n-1-i]`; needs negative-stride affine indexing or a **reverse** op
12. `compress` — prefix-count then place; needs **scan + scatter**
15. `bincount` — `out[a[i]] += 1`; needs **scatter** (values-as-indices) and integer index tensors
16. `scatter_add` — `out[link[j]] += values[j]`; the canonical **scatter**
17. `flatten` — reshape across rank; needs a **reshape / view** op

## Most impactful missing capabilities

1. **Gather / scatter (dynamic indexing)** — unlocks compress (12), bincount (15), scatter_add (16) and a gather route for roll (10); needed for embedding lookup, variable-length masking, and many real ML ops. Related: #293 (sharding & slicing).
2. **Scan / prefix-sum** — unlocks cumsum (7) and compress (12); needed for sequence processing.
3. **Reshape / view** — unlocks flatten (17).
4. **Reverse / modular affine indexing** — unlocks flip (11) and roll (10).

## Bonus finding (shape-inference bug)

The 1-D valid-convolution einsum `a +* "o<+k; k => o" kernel` over output-only axes **hangs**
OCANNL's shape inference (no error, just spins) even on a length-5 vector with a length-2 kernel.
The operator-matrix formulation of `diff` works around it; worth filing separately.
