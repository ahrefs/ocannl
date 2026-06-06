---
title: "Contraction Is Emergent: Projection Inference for OCANNL from Scratch"
author: "Łukasz Stafiniak and Claude (Anthropic)"
---

# Contraction Is Emergent

A companion piece, [*Broadcasting Is an Order*](https://lukstafi.github.io/notes/broadcast-aware-shape-inference.html), built OCANNL's shape inference from scratch: what a shape is, what it means for one shape to broadcast to another, and how a constraint solver fills in the unknown sizes. That account stopped where the sizes were known. But knowing the sizes is not enough to generate code. A tensor operation compiles to a loop nest, and before any loop can be written the compiler has to answer a different set of questions: *which* axes are driven by the same loop variable, *which* are summed away, *which* are read at a single fixed position no matter how the loops run. Sizes do not answer these. Two axes can have the same size and have nothing to do with each other; an axis can be summed without anyone ever writing the word "sum." This is the job of **projection inference**, and this post develops its core from scratch the same way the first developed shapes --- leaving out, for now, the two genuine complications (strided/convolutional indexing and axis concatenation), which get their own follow-up.

The thesis comes in two halves, and they are worth stating before any machinery, because the machinery is short once you have them.

The first half is a claim about *method*: **projection inference introduces no new constraints.** It re-reads the exact constraint set that shape inference already produced. Shape inference reads that set as an *order* --- sizes refining toward the specific, incompatible demands generalizing up toward a claim-free top. Projection inference reads the *same* set as an *equivalence* --- axes union-found into classes that share a loop variable. One elaboration, two algebraic readings. This is not merely economical (no second constraint generator to keep in sync); it is the load-bearing correctness argument. Indices that are derived from the same facts as the sizes cannot silently contradict the sizes. If projection inference invented its own constraints, the two could drift --- an index could iterate an axis the shapes say is size one, or fail to contract an axis the shapes made equal --- and the bug would surface only at runtime as a stride mismatch or a wrong answer. Sharing the source makes that class of bug unrepresentable.

The second half is a claim about *result*: **contraction is emergent.** Nowhere does a user, or the system, declare "reduce over this axis." A reduction is a *property of the inferred output index map* --- specifically, that the map does not mention some loop variable --- and it is read off after the fact, never written down. The matmul spec `a;b=>c` says which axes coincide; it never says which axis is summed. The sum happens because the coincident axis is *absent from the output*, so the loop that drives it writes the same output cell repeatedly, and repeated writes to one cell accumulate. We will make this precise, and it will turn out to be the cleanest thing in the system.

One tension stands between these two halves and the algorithm, and it is the only genuinely subtle point in the core: **co-iteration is strictly finer than size-equality.** Two axes can be forced to the same size by a constraint that has nothing to do with the operation we are compiling --- because they were contracted together *somewhere else* in the program. Shape inference, solving globally, happily identifies their sizes. Projection inference must not let that identification leak into *this* operation's indices. The resolution is a discipline we will name and then enforce: **global sizes, local identities.** It is the reason projection inference, despite reusing the constraints, re-derives and re-solves them per operation rather than reading answers off the global environment.

Let us build the target first, because projections are about a loop nest and the loop nest is the thing to hold in mind.

## What we compile to

Fix one tensor operation --- a single assignment, with a left-hand side (LHS) tensor written into and one or more right-hand side (RHS) tensors read from. Its compiled form has two parts.

The first is a **product space**
$$
P \;=\; R_1 \times \cdots \times R_m, \qquad R_j = \{0, 1, \ldots, n_j - 1\},
$$
a Cartesian product of integer ranges. Each factor $R_j$ has a **loop variable** $s_j$ ranging over it; iterating $P$ is iterating the nested loops $s_1, \ldots, s_m$ in some order. This is the iteration domain of the operation --- the set of points at which work happens.

The second is, for each participating tensor $T$, an **index map**
$$
\pi_T \;:\; P \longrightarrow \mathrm{Addr}(T)
$$
sending a point of the product space to a multi-index into $T$'s underlying array. We write $\pi_T(p) = (\pi_T^1(p), \ldots, \pi_T^{\,r}(p))$ for a rank-$r$ tensor, one component per axis. In the core fragment, each component is one of just two forms:

- $\mathsf{Iter}(s_j)$ --- this axis of $T$ is driven by loop variable $s_j$; as $s_j$ runs over $R_j$, this axis runs over its positions in lockstep.
- $\mathsf{Fix}(c)$ --- this axis is pinned to the constant position $c$, regardless of any loop. Position $0$ is the common case (a broadcast or a size-one axis); a non-zero $c$ comes from slicing.

(The full system has two more forms --- an affine combination $\sum_i c_i s_i + o$ for strides and convolutions, and a concatenation index --- but those are exactly the bells and whistles we are deferring. Their absence is what keeps everything below provably simple, and we will see precisely why.)

The operation's meaning is now a **map-reduce** over $P$. For each point $p \in P$:

1. read each RHS tensor $T_i$ at $\pi_{T_i}(p)$;
2. combine the read values with the operation's scalar function;
3. accumulate the result into the LHS at $\pi_{\mathrm{LHS}}(p)$.

"Accumulate" is the operative word, and it is where contraction lives. Consider a loop variable $s_j$ that the LHS map *does not mention* --- $\pi_{\mathrm{LHS}}$ has no $\mathsf{Iter}(s_j)$ in any component. Then as $s_j$ ranges over $R_j$ with the other variables held fixed, the LHS address $\pi_{\mathrm{LHS}}(p)$ does not change: every one of those $n_j$ points writes the *same* output cell. Writing the same cell repeatedly, under an accumulating operator, sums (or maxes, or whatever the accumulator is). So:

> **Reduction characterization.** A product axis $j$ is a *reduction axis* exactly when $\pi_{\mathrm{LHS}}$ is independent of $s_j$. There is no separate notion of "the reduced axes"; they are precisely the loop variables missing from the output index map.

This is the formal content of the second thesis: a reduction is read off $\pi_{\mathrm{LHS}}$, not written down.

Two derived properties of the index maps make the same fact actionable for the code generator, and OCANNL computes both:

- **Surjectivity** of $\pi_{\mathrm{LHS}}$ --- does every LHS cell get written by some product point? If not, the cells never addressed would keep stale contents, so they must be pre-set to the accumulator's neutral element.
- **Injectivity** of $\pi_{\mathrm{LHS}}$ --- does each LHS cell get written by at most one product point? If yes, no cell is ever accumulated into twice, so the read-modify-write can collapse to a plain write. If no, the accumulating read-back is required.

Injectivity fails exactly when a reduction axis exists --- a missing loop variable is what makes two distinct product points share an output cell. So the characterization is not a curiosity: it is the predicate that decides whether the emitted loop body is `lhs := f(rhs...)` or `lhs := lhs ⊕ f(rhs...)`.

The two properties are less independent than the split suggests, and the reason is that OCANNL's assignments are not in SSA form: an assignment reads and writes the same mutable tensor node, and the surface syntax controls the prior contents directly --- the colon in an accumulation like `=:+` (versus `=+`) erases the node to the neutral element before the map-reduce, while its absence accumulates into what was already there. So whether that pre-erase is needed turns on *both* properties at once: a cell must be cleared when it is either never written (non-surjective) or written more than once (non-injective --- the first accumulation would otherwise read an uninitialized cell). And the dual decision --- whether the accumulation can then collapse to a plain write, skipping the read-back --- arises only in the colon case, where injectivity settles it; without the colon the prior contents are deliberately kept and accumulated into, write-once or not. We return to the `=:+` syntax, and to where these assignments come from, at the end.

One structural invariant rounds out the target. A loop variable exists to traverse an axis, and an axis of size one has nothing to traverse --- its only position is $0$. So size-one axes never earn a product factor; they are $\mathsf{Fix}(0)$. Equivalently:

> **The product space ranges over exactly the axes of size greater than one** (excluding those pinned by a slice to a fixed position). Everything of size one is fixed at $0$.

With the target fixed, the question becomes: where do $P$ and the $\pi_T$ come from? From the constraints --- the same ones, read again.

## The same constraints, read again

Recall the constraint language from the shape post. Elaborating a program produces a set of atomic constraints over dimensions and rows, in two relations:

- **Equalities** $d_1 = d_2$ and $R_1 = R_2$, emitted by operations that demand exact matching --- einsum contractions, where an axis labeled `k` in the spec is *the same axis* as the operand axis at that position. No broadcasting.
- **Inequalities** $d_{\mathrm{result}} \sqsubseteq d_{\mathrm{operand}}$ and $R_{\mathrm{result}} \sqsubseteq R_{\mathrm{operand}}$, emitted by operations that permit broadcasting --- pointwise arithmetic, composition --- oriented so the more committed result refines the broadcastable operand and sits below it.

Shape inference takes these and computes sizes: equalities unify, inequalities accumulate as bounds, and the broadcast order's asymmetry (pins solve, permissive caps defer and may collapse up to the top) does the rest. Projection inference takes constraints of the *same form* and computes a loop nest instead --- "same form" rather than the same terms, because the dimensions here are not quite the ground terms of the shape algebra, for a reason worth disposing of first.

### Two equation forms

A ground dimension in the shape algebra is a size and a basis --- nothing more. That is all shape inference needed, but it is not enough here: two distinct axes of two distinct tensors can both be size $3$, and projection inference must be free to give them different array indices (they are different axes), so the map from a dimension to its projection term --- write it $\llbracket \cdot \rrbracket$ --- has to recover *which* axis a dimension is, and the size cannot tell it apart from any other size-$3$ axis. The dimensions projection inference works over therefore carry one extra component beyond size and basis: a **projection identifier**, a per-axis tag naming that axis instance. The identifier is the carrier of axis identity --- it, not the size, is what $\llbracket \cdot \rrbracket$ reads. (Where these identifiers come from, and why they are minted fresh, is the subject of the locality discipline below; for now, take it that every solved axis has one.)

With identity in hand, the re-reading produces **projection equations** in two forms:

- $\mathsf{Eq}(\rho_1, \rho_2)$ --- "$\rho_1$ and $\rho_2$ are the same projection": they share a loop variable (or are jointly pinned to the same fixed index). This is the co-iteration relation, and it is *symmetric*.
- $\mathsf{Iter}(\rho)$ --- "$\rho$ must be iterated": this axis needs a loop variable of its own even if nothing matches it.

The projection terms $\rho$ are minimal in the core: a solved axis becomes $\mathsf{Proj}(p)$, carrying its projection identifier $p$; an axis whose index the spec supplies independently of co-iteration becomes $\mathsf{Sol}(\mathit{idx})$. The $\mathsf{Sol}$ form is wider than just "size-one, read at $0$": its index can be a fixed constant (a size-zero or size-one axis, or a static slice picking a literal position) or an externally-bound iterator symbol (the runtime slice index of a `@|` batch-slice, supplied by an enclosing loop and passed in as a kernel argument). What unifies the $\mathsf{Sol}$ cases is that the operation neither infers the index nor ranges over it in its own product space --- the index comes from outside.

That is the whole core term language: $\mathsf{Proj}$ for an axis whose index this operation determines, $\mathsf{Sol}$ for one supplied from outside. There is no variable form. A spec label is settled one of two ways: if the spec supplies its index it is $\mathsf{Sol}$, and otherwise it labels a concrete operand axis and resolves to that axis's $\mathsf{Proj}$ --- by a mechanism we come to with the re-solve below. No core spec leaves a label undetermined, so nothing survives as an unknown. (The full system does carry a variable term, a binding map, and a deferred-iteration step --- but only to serve the strided and concatenated cases, where an axis is determined by a relation like $ih = \mathit{stride}\cdot oh + wh$ rather than by direct identity. That is derived-index machinery, and it goes with the follow-up.)

With $\llbracket d \rrbracket$ the projection term of a dimension $d$, the elaboration is six rules. They are the whole of core projection inference.

**1. Dimension equality.** $\;d_1 = d_2 \;\rightsquigarrow\; \mathsf{Eq}(\llbracket d_1 \rrbracket, \llbracket d_2 \rrbracket)$.

Einsum's exact match becomes co-iteration: two axes the spec labeled alike are driven by one loop variable. This is the engine of contraction --- it is what makes the two `k` axes of a matmul share a variable.

**2. Dimension inequality, broadcasting operand.** A broadcast inequality whose operand-side axis is size one, $d_{\mathrm{result}} \sqsubseteq (d_{\mathrm{operand}} = 1)$, elaborates to
$$
\mathsf{Eq}(\llbracket d_{\mathrm{operand}} \rrbracket, \mathsf{Sol}(\mathsf{Fix}(0))).
$$
The broadcasting axis is *pinned to position $0$*: it is read at its single cell for every value the larger axis takes. Crucially it is **not** unified with the larger axis. This is the cleanest sentence-level link between the two passes: in shape land a broadcast inequality is an *order* fact (the size-one unit sits above the specific axis, which it broadcasts to fill); in projection land the same inequality *severs* (the small axis is cut loose and read at $0$ while the large one iterates alone). **Equality unions; broadcasting severs.** An axis that broadcasts is precisely an axis that does not get its own loop --- it rides along, fixed, while its partner drives.

**3. Dimension inequality, otherwise.** $\;d_1 \sqsubseteq d_2 \;\rightsquigarrow\; \mathsf{Eq}(\llbracket d_1 \rrbracket, \llbracket d_2 \rrbracket)$.

When neither side is a broadcasting unit --- two genuine, equal, iterated axes related by a pointwise inequality --- they co-iterate, exactly as an equality would have it. So the inequality path and the equality path agree wherever broadcasting is not actually happening; they diverge only at rule 2, the genuine broadcast.

**4. Row equality.** $\;R_1 = R_2$: align the two rows at their outer edges --- leading flank against leading flank from the front, trailing against trailing from the back --- and emit a dimension equality (rule 1) per aligned pair. No broadcasting; the ranks must already match.

**5. Row inequality.** $\;R_{\mathrm{result}} \sqsubseteq R_{\mathrm{operand}}$: align the flanks as above, but now the larger row (the result) may out-rank the smaller. Its surplus interior axes --- the ones the operand side does not reach --- each emit $\mathsf{Iter}$: they iterate on their own, broadcast targets with no partner below. Each aligned pair emits a dimension inequality (rules 2--3); and at any pair where the operand axis is the size-one broadcaster, the pair is rewritten to "*result iterates, operand pinned to $0$*" --- the row-level form of rule 2's severance.

**6. Terminal markers.** A *terminal* axis --- one belonging to a leaf tensor, or to an RHS whose axes might otherwise go unmatched --- emits $\mathsf{Iter}(\llbracket d \rrbracket)$ for each of its axes.

Rule 6 deserves its own sentence, because without it the system would be silently incomplete. Rules 1--5 all relate *pairs* of axes. But an axis can exist without a partner: assign a constant into a result and the result's axes have nothing on the RHS to match; a leaf parameter read in isolation has no other operand. Such an axis would acquire no loop variable, and would never be traversed. The terminal markers are the floor that guarantees every axis that genuinely exists gets visited: applied to a size-one axis the marker is a harmless no-op (size one needs no loop), but applied to a real axis it forces a fresh iterator. Together with the size-one invariant, this gives exactly the product space we wanted --- one factor per axis of size greater than one.

### Global sizes, local identities

Now the subtlety. The constraints we are re-reading were solved *globally* by shape inference: a single environment, accumulated over the entire program, that may have identified the sizes of two axes because they met in *some other* operation. Suppose axes $\alpha$ and $\beta$ both feed, elsewhere, a reduction that forces them to the same size $n$. Shape inference records $\alpha$ and $\beta$ as size $n$ --- correctly. But in the operation we are now compiling, $\alpha$ and $\beta$ might be entirely independent axes that simply happen to be size $n$. If projection inference inherited the global identification, it would conclude $\alpha$ and $\beta$ co-iterate here, fuse their loops, and generate wrong code: an elementwise pairing where there should have been an independent double loop.

This is why co-iteration must be *finer* than size-equality, and why it cannot be read off the global environment. The enforcement is a three-step discipline, run *per operation*:

1. **Freshen** every projection identifier in the operation's (already size-solved) shapes, so that no $\mathsf{Proj}$ id is shared with any other operation.
2. **Re-derive** this operation's constraints --- and only this operation's --- from those freshened shapes.
3. **Re-solve** them in a *local*, initially empty environment, so the resulting co-iteration classes contain only identifications this operation actually forced.

Step 3 is easy to mistake for a no-op --- every tensor shape is closed before projection inference runs (a hard precondition, the analogue of shape inference's "close before you read"), so what is left to solve? The answer is the einsum spec. Re-deriving the constraints matches the operation's spec against its operands, and the spec is a shape-shaped template carrying fresh variables of both sorts: its ellipsis marks are *row* variables, and its axis labels are *dimension* variables, both minted anew on each derivation. The two are not on the same footing, though, and the difference is what makes the re-solve necessary rather than merely convenient.

A fresh *dimension* variable need not be solved here at all: a spec label that the re-solve leaves undetermined could simply be carried into projection inference as a projection variable and resolved there, by the same union-find that resolves everything else --- a dimension unknown is exactly the kind of thing the projection solver could absorb. A fresh *row* variable cannot be deferred that way. Until the rows are aligned --- until it is settled which axes of which operand the template's open stretch covers --- there are no per-axis pairings to hand to projection inference at all; the row structure has to be fixed *before* the projection equations can even be formed. So the row variables are the load-bearing reason for the re-solve, and the dimension labels come along for the ride: the same alignment that pins the rows also unifies each label with the operand axis it matches, so a label `i` becomes that concrete, already-sized axis and inherits its identifier.

The upshot is that every fresh dimension variable is in fact *eliminated* by the time the solve finishes --- a spec label always lands on some operand axis, so it ends up a solved dimension carrying that axis's identifier rather than a leftover unknown. This is why the core term language needs no variable form: not because a dimension variable could not have survived in principle, but because the row-alignment that the re-solve must do anyway disposes of them as a side effect. The variables exist during the solve and are gone by the time the elaboration reads the result.

So the sizes are taken as given (global, already settled) while the identities are recomputed from scratch (local, this operation only). **Global sizes, local identities.** It looks like duplicated work --- we solved these constraints once already --- but it is not the same work: the first solve answered "how big?", globally, and this one answers "which loop?", locally, and the second answer is genuinely unavailable from the first. The freshening is what discards the over-identification; re-deriving from the same constraint *forms* is what keeps the indices faithful to the shapes. And the freshening is also the step that made $\llbracket \cdot \rrbracket$ well-defined: stamping each axis of this operation with its own identifier is what gives every dimension the identity the elaboration reads. Well-definedness of the elaboration and locality of the identities are thus one fact, not two --- the per-operation fresh identifier is the carrier of both. Both halves of the opening thesis are in this one move.

## Solving: union-find with pinning

With the equations in hand, solving is a single pass over them maintaining three pieces of state:

- a **union-find** structure `classes` over projection ids, whose find-representative answers "which co-iteration class?";
- a **pin map** `pins` from a class to a concrete `axis_index`, for classes forced to a literal position;
- an **iterate set** `I` of classes that must receive a loop variable.

The pass interprets each equation:

- $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Proj}\,q)$ --- union the classes of $p$ and $q$. The two axes carry their sizes; if those disagree, fail (this is a real bug --- two axes asserted to share a loop but of different lengths). Otherwise record the class's size.
- $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Sol}(\mathit{idx}))$ --- pin the class of $p$ to $\mathit{idx}$. If it is already pinned to a different index, fail.
- $\mathsf{Eq}(\mathsf{Sol}\,i, \mathsf{Sol}\,j)$ --- check $i = j$; fail if not. (No class is created; two literals either agree or they do not.)
- $\mathsf{Iter}(\mathsf{Proj}\,p)$ --- add the class of $p$ to $I$.
- $\mathsf{Iter}(\mathsf{Sol}\,\_)$ --- a no-op (an externally-supplied index does not iterate in this operation).

When the pass finishes, **close** each class:

- if it is pinned, it takes its pinned index --- and *pinning dominates iteration*: a class both pinned and in $I$ keeps the pin (a sliced or broadcast axis is read at its position, not looped, even if some marker also asked it to iterate);
- otherwise, if it is in $I$ *and* its size is greater than one, it gets a **fresh iterator symbol** --- a new loop variable, joining the product space;
- otherwise (unpinned, and either not iterated or size one), it needs no loop variable and reads as $\mathsf{Fix}(0)$.

That is the entire solver. The product space is read off as one factor per fresh iterator (with its recorded size); each tensor's index map is read off by mapping each axis to its class's closed index.

The contrast with shape closing is sharp and instructive. Shape inference's closing phase must *choose a direction* --- leaves commit downward to their most specific fitting shape, interiors commit upward to the top --- because the broadcast order has a meaningful top to default to and a leaf/interior distinction that decides which way to fall. It can even *collapse* two incompatible caps to the top, a semantically loaded resolution that turns a would-be error into a broadcast. Projection closing does none of this. It chooses no direction, defaults to no pole, collapses nothing. It only *labels*: pinned classes wear their pin, iterating classes get a fresh name, the rest are $\mathsf{Fix}(0)$. An equivalence has no up or down --- union is symmetric --- so there is no policy to state, only a supply of fresh symbols. The order-theoretic pass needs a closing *policy*; the equivalence-theoretic pass needs only a *naming*.

## Why the core is canonical

The reward for deferring the bells and whistles is a clean canonicity result.

Fix one operation and its finite set $E$ of projection equations. Define:

- $\approx$ as the least equivalence relation on projection ids containing every pair $(p, q)$ with $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Proj}\,q) \in E$;
- the pin map as the partial function induced by the equations $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Sol}(\mathit{idx}))$, lifted to $\approx$-classes;
- the iterate set as the $\approx$-closure of the classes named by $\mathsf{Iter}$ markers.

> **Canonicity.** The solution is forced and unique up to renaming of the fresh iterator symbols. There is no guessing and no closing direction.

Each ingredient is forced. The relation $\approx$ is a *closure* --- the least equivalence containing the equated pairs --- which union-find computes directly, independent of the order the equations arrive in. The pin map is a *checked partial function*: each $\mathsf{Eq}(\cdot, \mathsf{Sol})$ either extends it consistently or contradicts it, and the $\mathsf{Sol}$-vs-$\mathsf{Sol}$ equations are pure checks; neither selects among alternatives. The iterate set is again a closure. The labeling is then a total function of these three: pin if pinned, else fresh iterator if iterating and size $> 1$, else $\mathsf{Fix}(0)$. The only freedom anywhere is the *identity* of the fresh symbols --- renaming, the same caveat the shape post made for its fresh template variables.

So core projection inference is, in a precise sense, *more* principal than core shape inference: shape inference resolves genuine underdetermination at closing (which way does a free variable fall?), whereas projection inference only succeeds or reports a conflict --- it never resolves an ambiguity because it never faces one.

The follow-up will spend exactly what this scoping bought. The affine/convolution and concatenation constructs we deferred are precisely the ones whose projection terms are *derived indices* --- an input axis indexed by `stride · output + dilation · kernel − padding`, or an axis indexed by whichever of several concatenated components is active. A derived index cannot be settled by union-find alone, because its value depends on the loop variables of *other* classes, which are not known until those classes are themselves solved. Reintroducing them forces a *stratified* solver --- first fix the base co-iteration classes, then *evaluate* the derived terms against them, with deferred equations that can only be checked once their inputs are solved --- and with stratification comes the possibility of late-discovered conflicts and a genuine product-space coupling (concatenated components must iterate sequentially within one factor rather than as independent loops). The core, having no derived indices, is a flat union-find with a labeling.

## Three worked examples

A single thread runs through these: equality unions, broadcasting severs, and contraction is whatever the output forgot to mention.

### Matrix multiplication

The einsum `a;b=>c` over `a[i,j]`, `b[j,k]`, `c[i,k]` emits, by rule 1, three dimension equalities --- the `i`s match, the `j`s match, the `k`s match:
$$
\mathsf{Eq}(c_i, a_i), \quad \mathsf{Eq}(a_j, b_j), \quad \mathsf{Eq}(c_k, b_k).
$$
Union-find produces three classes, $I = \{c_i, a_i\}$, $J = \{a_j, b_j\}$, $K = \{c_k, b_k\}$, all iterated (all sizes $> 1$), so each gets a fresh iterator: $i^\star, j^\star, k^\star$. The index maps fall out by reading each axis's class:
$$
\pi_c = (\,i^\star,\ k^\star\,), \qquad \pi_a = (\,i^\star,\ j^\star\,), \qquad \pi_b = (\,j^\star,\ k^\star\,).
$$
Now apply the reduction characterization. The product space is $\{i^\star, j^\star, k^\star\}$, but $\pi_c$ mentions $i^\star$ and $k^\star$ and *not* $j^\star$ --- so $j^\star$ is a reduction axis, though the spec named three coincidences and never a sum. As $j^\star$ runs, $\pi_c$ stays put, the same `c[i,k]` cell is written for every `j`, and the accumulator adds them. $\pi_c$ is non-injective, which is the predicate that tells the code generator to emit the accumulating read-modify-write.

### Pointwise broadcast

Multiply a scalar `s` by a matrix `m[p,q]` into `r[p,q]`. The pointwise operation emits inequalities. The inequality $r \sqsubseteq m$ matches axis for axis (rule 3): $\mathsf{Eq}(r_p, m_p)$ and $\mathsf{Eq}(r_q, m_q)$ --- co-iteration, classes $P = \{r_p, m_p\}$ and $Q = \{r_q, m_q\}$. The inequality $r \sqsubseteq s$ is the interesting one: `s` is a scalar, so against `r`'s two axes its sole content is broadcasting units, and rule 5 (with rule 2 at each pair) rewrites to *super iterates, sub pinned to $0$* --- the `r` axes are already iterating via $P, Q$, and `s`'s axes are pinned:
$$
\pi_r = \pi_m = (\,p^\star,\ q^\star\,), \qquad \pi_s = (\,\mathsf{Fix}(0),\ \mathsf{Fix}(0)\,).
$$
Note what did *not* happen: `s`'s axes were not unioned with $P$ or $Q$ --- broadcasting severed them and fixed them at $0$, so the scalar is read at one cell for every $(p^\star, q^\star)$ while the matrix and result iterate together. Every product axis appears in $\pi_r$, so $\pi_r$ is injective and surjective: no reduction, no zero-init, a plain elementwise write.

### Summing by a vector of ones

The shape post described a deliberate idiom: to sum out an axis, multiply by a ones-vector whose length is left to inference, which then closes downward to match the axis being reduced. Look at the same idiom through the projection lens. To sum axis `q` of `m[p,q]` into `r[p]`, multiply `m` by `ones[q]` and let the contraction land. The constraint that `ones`'s axis matches `m`'s `q` axis emits (rule 1) $\mathsf{Eq}(\mathit{ones}_q, m_q)$ --- they co-iterate, class $Q$. The result `r` has no `q` axis, so $\pi_r = (p^\star)$ mentions $p^\star$ but not $q^\star$, and $q^\star$ is a reduction axis by the characterization. Same idiom, both readings: shape inference *grew* the ones-vector to the right length (closing downward from use), and projection inference *summed over it* (the output omits `q`). The one constant the user declined to size is sized by the order; the one sum the user never wrote is induced by the absence. The two passes meet on the same line of code, each doing its half, neither needing a fact the other did not already imply.

## From what you write to the projections

None of the machinery above is what an OCANNL user writes, and it is worth closing the loop to the surface, because the loop closes onto two distinct things --- and projections live on exactly one of them.

OCANNL has two intermediate representations. The high-level one is **assignments**: the accumulating statements `lhs <accum-op> f(rhs…)` we have been compiling, each carrying the `projections` record this article derived. The lowered one is a for-loop language the assignments compile down to, and it is not our concern here --- by the time it exists the projections have been spent, discharged into actual loops and indexed accesses. Projections are a property of the *high-level* IR: one record per assignment, the thing that says how that assignment's loops run and how its tensors are indexed. Assignments are what the `%cd` ("code") syntax builds, and the `=:+` from the reduction discussion was already a glimpse of it --- the colon erasing, the `+` accumulating.

Sitting above both IRs is the tensor frontend. A `Tensor.t` is not a third IR; it is a first-class node in a differentiable expression graph, bundling the assignment-level forward and backward code with the shape that inference fills in. Tensors are ordinary OCaml values: with the right module open, `a * b` is just an infix operator on `Tensor.t`, and the `%op` ("operation") syntax is sugar over that --- it adds conveniences like tensor literals and inline tensor definitions, but the expression graph is plain OCaml underneath. When a user writes `a * b`, they are building tensors; the assignments --- and their projections --- are generated underneath.

The bridge between the two is where projection inference does its work. A tensor operator is defined by handing the frontend two `%cd` functions, an `op_asn` for the forward pass and a `grad_asn` for the backward, each of type roughly $(\ldots) \to \texttt{projections} \to \texttt{assignments}$: given inferred projections, emit the assignment that uses them. Matrix multiplication is, stripped to its core,
$$
\begin{aligned}
\texttt{op\_asn} \;&=\; (\texttt{v} \mathrel{\texttt{=:+}} \texttt{v1} \mathbin{\texttt{*}} \texttt{v2}), \qquad \texttt{compose\_op} = \texttt{Compose}, \\
\texttt{grad\_asn} \;&=\; (\texttt{g1} \mathrel{\texttt{=+}} \texttt{g} \mathbin{\texttt{*}} \texttt{v2}; \;\; \texttt{g2} \mathrel{\texttt{=+}} \texttt{v1} \mathbin{\texttt{*}} \texttt{g}).
\end{aligned}
$$
The user writes `a * b`; `compose_op` selects the compose path, which (as the worked example traced) emits the constraints whose projection reading is the `a;b=>c` loop nest; that derived `projections` record is handed to `op_asn`, which produces the single assignment `v =:+ v1 * v2` carrying it. The `=:+` is now fully cashed out: the `+` is the accumulation that sums the contracted `j`, and the `:` pre-erases because $\pi_c$ is non-injective --- exactly the predicate from the reduction discussion, decided by the projections this article derived. The user wrote a product; the contraction, the loop nest, and the pre-erase were all inferred.

The gradient assignments are where the colon's *absence* earns its keep. Each backward statement reads `=+`, not `=:+`: it accumulates the incoming gradient `g` into the operand gradients `g1`, `g2` *without* erasing them first. The reason is exactly what the colon controls --- an operand can be used in more than one place in the surrounding computation, and its gradient must sum the contributions from every use, so the backward pass adds to whatever is already there rather than overwriting it. Forward computes `c` afresh and so erases (`=:+`); backward contributes to a running total and so preserves (`=+`).

The projections, though, are not simply shared across these assignments --- each assignment has its own `projections` field, *derived* from the one the operation inferred. There is a single inferred record --- the operation's, the `a;b=>c` product space and its three per-tensor index maps from the worked example --- and it is the `op_asn`'s projection, the original. Every other assignment's projection is computed from it by a *slot transformation* keyed to how its tensors are named. The `%cd` syntax reads an identifier's name as a slot: `v`, `t`, `g` (and `lhs`) name the LHS slot; `v1`, `t1`, `g1` the first RHS slot; `v2`, `t2`, `g2` the second; and each slot says which of the original record's maps that tensor should be indexed by. So in `g1 =+ g * v2`, the assignment's *own* left-hand side is `g1` --- but `g1` carries the RHS1 slot, so it is indexed by the original's RHS1 map, the very $(i,j)$ that operand `a` used going forward; `g` reads the LHS map $(i,k)$, and `v2` the RHS2 map $(j,k)$. Each gradient assignment is the forward operation's index maps re-addressed: the same three maps, the same product space $\{i,j,k\}$, only with a different one of them serving as the LHS. And that is what moves the contraction. The forward LHS map $(i,k)$ omitted $j$, so forward contracted $j$; `g1`'s LHS map is $(i,j)$, which omits $k$, so it contracts $k$ --- matching $\partial a = \sum_k g_{ik}\,b_{jk}$. The companion assignment `g2 =+ v1 * g` makes `g2` (RHS2 slot) the LHS, map $(j,k)$, which omits $i$, so it contracts $i$ --- matching $\partial b = \sum_i a_{ij}\,g_{ik}$. The reduction characterization does all three: in each case the contracted axis is simply the product axis the LHS map leaves out.

This is why a gradient needs no shape or projection inference of its own. The roles are permuted, but the maps are not recomputed --- they are read out of the one record by slot. One inference; many assignments, each a slot-permuted view of it. (The same naming convention does heavier lifting when a gradient needs an *intermediate* tensor of a particular shape --- a max-pool or tropical-convolution backward must record which input position won the argmax, which is an input-shaped fact, so the intermediate is named with an `_rhs1` suffix to claim the input slot rather than the output one. That is where slot detection stops being bookkeeping and starts being load-bearing, and it belongs with the follow-up.)

Everything so far has been the *consumer* regime: an assignment written inside an operator definition, where a `projections` record is already in context and the assignment reads a slot-permuted view of it. The tensor nodes there were introduced by tensor-expression construction (`a * b`), so the shape and projection inference was driven by that construction, and the assignments only consume its result. But assignments are also written *outside* operator definitions, with no ambient `projections` --- and there each assignment is a *producer*: it contributes its own constraint to shape and projection inference, exactly as a tensor-expression operator would, only written directly at the assignment level. How it contributes is set by a `~logic` annotation: `"."` for pointwise, `"@"` for compose, `"T"` for transposing a unary op's input against its output --- all three taking the broadcast *inequality* path --- and any other string read as a full einsum spec, the *equality* path. With neither a `projections` in context nor a `~logic` given, the contribution defaults to pointwise `"."`, with a single exception for `*` that we come to below.

The canonical producer case is an optimizer update. An SGD step over a parameter `p`, simplified to its assignment skeleton, is a sequence of standalone assignments:

```ocaml
{ sgd_delta } =: p.grad + (!.weight_decay *. p);
{ sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta;
p =- learning_rate *. sgd_delta
```

None of these sits inside an operator definition; there is no inherited `projections`, so each contributes its own constraint to inference. Each assignment combines an accumulation operator (the `=:` clearing-then-writing, the `=-` subtracting into the target) with a primitive operation applied to its right-hand side --- and in the first two lines that primitive is the `+` of `Add`, taking two operands. Where a `*.` appears it is something else: not a recognized assignment primitive but the pointwise tensor multiply, so it introduces a tensor sub-expression as an argument, and that sub-expression's own construction drives shape and projection inference. The last line repays a closer look, because the same scaling can be written two structurally different ways. As written, `p =- learning_rate *. sgd_delta` is a *unary* assignment: `*.` is the pointwise tensor multiply, so `learning_rate *. sgd_delta` is a tensor sub-expression that builds an intermediate, and the assignment merely accumulates that one intermediate into `p`. Written instead as `p =- learning_rate * sgd_delta ~logic:"."`, it is a *binary* assignment: `*` is the assignment-level arithmetic primitive, with two right-hand operands and no intermediate, and the `~logic:"."` makes the accumulation pointwise. The two optimize to identical code once the intermediate is inlined, but they are different assignments --- one unary over a subexpression, one binary over a primitive.

The `~logic:"."` is needed only in the binary spelling, and the reason is an overload of `*` it would be nicer not to have. The symbol `*` is context-dependent: in an assignment's outermost operator position it is the arithmetic primitive `Mul`, but everywhere else --- in a tensor expression --- it is *compose*. (Its pointwise sibling `*.` has no such ambiguity; it is a tensor pointwise operation in every context.) So when `*` appears as the binary operator of an assignment, nothing about it pins the projection logic --- its tensor-expression reading would be compose, not pointwise --- and rather than let it default surprisingly it is left undetermined, to be told `~logic:"."`. This is a known rough edge of the surface syntax, not a designed feature; it is the single exception to the `"."` default. And `sgd_delta` and `sgd_momentum` are not pre-existing tensors: the `{ … }` braces are inline definitions, introducing fresh intermediate nodes whose shapes are *inferred* from the very constraints these assignments contribute. So the producer regime does both jobs at once --- it drives inference for the assignment's existing operands *and* shapes the new tensors the assignment introduces.

This same update is where the predecessor's opening scalar lives: `learning_rate *. sgd_delta` is the learning-rate scalar scaling the parameter delta, broadcasting against it exactly as the shape article's opener described. Its projection reading is the pointwise severance of the worked example --- the scalar pinned to $\mathsf{Fix}(0)$, the delta's axes iterating. The scalar that opened the shape article is a worked instance of this one's broadcast rule.

The other place projections reach the surface is the einsum spec notation that drives the strided and concatenated cases, and it too belongs with the follow-up.

## Where this leaves us

We took the loop nest as the thing to explain --- product space and index maps, with reduction defined as the output's independence of a loop variable --- and derived it from the constraints shape inference had already produced. Reusing that one elaboration, read as an order for sizes and as an equivalence for loops, is what both economizes the system and guarantees the indices respect the shapes; the locality discipline (global sizes, local identities) is what keeps coincidences of size elsewhere from fusing loops here.

The deferred constructs --- strided and convolutional indexing, and axis concatenation --- are exactly the derived-index machinery whose absence made the core a flat, canonical union-find. The follow-up reintroduces them, and with them the stratified solve-then-evaluate, padding-as-fixpoint, and product-space coupling that derived indices demand. The core developed here is what those extensions rest on.

OCANNL is open source, at [`github.com/ahrefs/ocannl`](https://github.com/ahrefs/ocannl).
