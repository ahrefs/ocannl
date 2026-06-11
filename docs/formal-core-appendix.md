# The Formal Core of OCANNL — Proof Appendix

Companion to `docs/blog/ocannl-formal-core.md` (referred to as *the core*). The core keeps statements, proof ideas, status labels, and the running implementation audit; this appendix holds the full write-outs, keyed to the core's numbering. Notation is the core's throughout: $1_\emptyset$ the claim-free unit, $\sqsubseteq$ the broadcast order (lower = more material), $\diamond$ the marker, $\approx$ the flat equivalence of Def. 2.9 and the satisfaction relation of row-equality constraints (Def. 3.4), $\sqsubseteq^1$ the one-shot order ($X \sqsubseteq^1 Y \iff \mathrm{flat}(X)^\bullet \sqsubseteq Y$).

Status of the core's obligations after this appendix:

| Obligation | Status |
|---|---|
| Lemma 5.1 assembly (per-rule, semantic/policy split) | **proved** (A.1) |
| Def. 4.3(c) split-surplus case | **corrected** — the calculus now defers, matching the implementation (A.1.3) |
| Lemma 5.2.2 rank conservativity, rule by rule | **proved** (A.2) |
| Thm. 5.2(a) lineage bookkeeping bound | **proved** (A.3) |
| Confluence across fair strategies (Thm. 5.6 caveat) | **discharged by decision** — strategy fixed (A.4) |
| Prop. 6.2 re-solve termination case split | **proved, strengthened** — re-solves always terminate (A.5) |
| Prop. 6.4(iii) $\sqsubseteq^1$-greatestness | **proved**, with a rank-policy qualification (A.6) |
| Remark 6.7(i) order-robustness of closing | **proved** at dimension sort (A.7) |
| Remark 2.15(iv) inference-policy-vs-checking | **proved** (A.8) |
| Detection Lemma (Thm. 5.2(b)) | **open** — structured reduction, D1 proved, D3 the residual gap (A.9) |
| Marker-provenance invariant; surface discharge | **open** — statement recorded (A.10) |
| Lemma 7.7 elaboration bookkeeping vs `shape.ml` | **open** — implementation-facing, deferred |
| §8 conjectures (8.2, 8.3, 8.4) | **open** — staged-extension work, deferred |

---

## A.1 Lemma 5.1 (Solution preservation, with a policy coordinate) — full proof

Recall the denotation $\widehat{C} = \Phi \cup \mathrm{eqns}(\sigma) \cup \mathrm{constr}(B)$, with $\mathrm{eqns}(\sigma)$ read as $\approx$-constraints at row sort. We must show, per rule: *semantic steps* preserve $\mathrm{Sol}(\widehat{C})$ exactly up to extension on fresh variables, and fail only when $\mathrm{Sol}(\widehat{C}) = \emptyset$; *policy steps* are sound (new solutions restrict to old) and lose only solutions that some policy commitment forecloses.

Throughout, "the rule preserves $\mathrm{Sol}$" means: $\widehat{C}_{\mathrm{old}}$ and $\widehat{C}_{\mathrm{new}}$ have the same solutions after restricting the new ones to the old variables, and every old solution extends. Substitution steps ($\sigma := [x \mapsto T]\circ\sigma$ with re-emission) preserve the denotation *literally*: the binding moves $x = T$ from $\Phi$ into $\mathrm{eqns}(\sigma)$, the re-emissions move each stored bound from $\mathrm{constr}(B)$ back into $\Phi$ instantiated at $T$ — which is the same constraint modulo the equation $x = T$ that is itself in the denotation — and substitution through $\Phi$ rewrites constraints modulo the same equation. So for every rule it suffices to analyze the *consumption* of the triggering constraint.

### A.1.1 Dimension rules (Def. 4.2) — all semantic

- **DE-refl, DI-top, DI-refl**: the consumed constraint is valid ($t = t$; $d \sqsubseteq 1_\emptyset$ by Def. 1.2; reflexivity); removing a valid constraint preserves $\mathrm{Sol}$.
- **DE-clash, DI-ground (failing case)**: the consumed constraint is unsatisfiable on its own (two distinct ground atoms are never equal; Def. 1.2 fails); $\mathrm{Sol}(\widehat{C}) = \emptyset$.
- **DE-bind**: replaces the constraint $\alpha = t$ by the binding; preservation is the substitution discipline above. The occurs concern is vacuous: dimension terms in the core fragment are atoms or variables.
- **DI-pin** ($\alpha \sqsubseteq d$, $d$ an atom): by Def. 1.2, $e \sqsubseteq d$ iff $e = d$. So the constraint is *equivalent* to $\alpha = d$: same solution set, and the replacement is an entailed equality.
- **DI-pin-top** ($1_\emptyset \sqsubseteq \alpha$): by Def. 1.2, $1_\emptyset \sqsubseteq e$ iff $e = 1_\emptyset$. Same form as DI-pin.
- **DI-cap**: recording moves the constraint into $\mathrm{constr}(B)$ — denotation unchanged. Discarding a duplicate removes a constraint implied by an identical one. The collapse ($\mathrm{lb}(\alpha) = d' \ne d$): the denotation contains $d \sqsubseteq \alpha$ and $d' \sqsubseteq \alpha$ with $d \ne d'$ atoms; by Prop. 1.3 the join of two distinct atoms is $1_\emptyset$, and the only elements above both are upper bounds of the join, i.e. $\{1_\emptyset\}$; so both constraints together are equivalent to $\alpha = 1_\emptyset$, an entailed equality.
- **DI-adj**: deferral into $B$ — denotation unchanged.

### A.1.2 Row equality (Def. 4.3, cases a, b, d)

Fix the constraint $l_1 \cdot \langle \rho_1 \rangle \cdot r_1 \approx l_2 \cdot \langle \rho_2 \rangle \cdot r_2$ (degenerate row-variable-free sides allowed). Its ground meaning under Def. 3.4 is the word equation $\gamma l_1 \cdot \mathrm{flat}(\gamma\rho_1) \cdot \gamma r_1 = \gamma l_2 \cdot \mathrm{flat}(\gamma\rho_2) \cdot \gamma r_2$ in $D^*$ — flat, marker-blind.

**(a) Both sides closed.** The word equation between two ground-length words holds iff lengths (ranks) agree and the flat zip holds pointwise. The rule fails exactly on rank disagreement and otherwise emits exactly the zip: equivalent reformulation, semantic. (The former marked-equality clause "fail on marker mismatch" rejected $\approx$-satisfiable instances; it is gone, and the implementation never had it — `unify_row`'s closed–closed path compares flat.)

**(b) One side open, the other closed.** Write the open side $l \cdot \langle\rho\rangle \cdot r$, the closed content $w = \mathrm{flat}(C)$. The word equation is $\gamma l \cdot x \cdot \gamma r = w$ with $x = \mathrm{flat}(\gamma\rho)$. Since $|\gamma l| = |l|$ and $|\gamma r| = |r|$ are known, solutions exist iff $|l| + |r| \le |w|$ — the total-overflow complement — and then $x$ is *uniquely* the middle $m$ of $w$, with $\gamma l$, $\gamma r$ zipped against $w$'s outer edges. So the rule's failure condition and its emitted dimension equalities and entailed flat content are exact: semantic. The *binding* $\rho \mapsto m_l \cdot \diamond \cdot m_r$ additionally selects a placement among the $|m| + 1$ split positions, which the $\approx$-semantics leaves free and Prop. 2.10(ii) makes pairwise $\sqsubseteq$-incomparable: the placement coordinate is the policy step, sound (the committed representative is a genuine choice of $x = m$), and lossy exactly when a stored *marked* inequality discriminates the placement (Remark 2.11). *(The implementation's asymmetric trailing guard — fail when $|r| > |r_C|$, the closed side's structural trailing flank — is a conservative deviation: it rejects some instances with $|l| + |r| \le |w|$ that are $\approx$-satisfiable. Being placement-sensitive on the closed side's split, it is a policy-rejection site; pinned by the mirror-orientation case in `test/einsum/test_row_self_reference.ml`.)*

**(d) Same variable both sides.** After cancelling the aligned overlaps the residue is $x \cdot t = s \cdot x$ with $s, t$ the surplus words and $x = \mathrm{flat}(\gamma\rho)$. *Unequal surplus totals*: lengths give $|x| + |t| \ne |s| + |x|$, no solution — the occurs failure is semantic. *Equal totals* (rotational): by Lyndon–Schützenberger, solutions are exactly $x \in (uv)^* u$ where $s = uv$, $t = vu$ ranges over the conjugacy splittings; the deferral leaves the constraint in flight (denotation unchanged — semantic as a step), and the *eventual upward close* at the guessing stage commits $x = [\,]$, which is a solution iff $s = t$: a policy step, sound, and lossy exactly on the nontrivial conjugate instances. This is the explicitly accepted incompleteness of Def. 4.3(d).

### A.1.3 Row equality, case (c) — both open, distinct variables: **the corrected rule**

Cancel the aligned overlaps (outer-edge zips — semantic, as in (b)). Two configurations remain.

**(c-nested) Both surpluses on one side** — say side 2's term carries surplus flanks $m_l$ (leading) and $m_r$ (trailing) relative to side 1's. The residue is $x_1 = m_l^\gamma \cdot x_2 \cdot m_r^\gamma$ — for *every* grounding, by cancelling the common outer flanks of the flat words. The content of the binding $\rho_1 \mapsto m_l \cdot \langle\rho_2\rangle \cdot m_r$ is therefore entailed: semantic, with the placement-inheritance policy coordinate exactly as in (b).

**(c-split) Surpluses split across the sides** — side 1 in surplus on one flank, side 2 on the other. After cancellation the residue is the **two-variable word equation**
$$s \cdot x_1 = x_2 \cdot t, \qquad |s|, |t| > 0,$$
($s$ side 1's leading-flank surplus, $t$ side 2's trailing-flank surplus, in the orientation of the pinning test; the mirror orientation is symmetric). Its solution set over $D^*$ is:

- the **principal family** $x_2 = s \cdot w,\ x_1 = w \cdot t$ for arbitrary $w \in D^*$; and
- the **sporadic solutions**: $x_2$ a *proper prefix* of $s$, say $s = x_2 \cdot u$ with $u \ne \varepsilon$, and then $u \cdot x_1 = t$, i.e. $u$ a prefix of $t$ and $x_1$ its complement — at most $|s|$ of them, one per proper-prefix split of $s$ that matches $t$.

*(Proof of the dichotomy: compare $|x_2|$ with $|s|$. If $|x_2| \ge |s|$, then $s$ is a prefix of $x_2$, giving the family; if $|x_2| < |s|$, then $x_2$ is a proper prefix of $s$ and the rest is forced. The two cases are exhaustive and the parameters are free exactly as stated.)*

A fresh-variable binding $\rho_2 \mapsto s\cdot\langle\rho'\rangle$, $\rho_1 \mapsto \langle\rho'\rangle\cdot t$ — the rule as previously written in the core — captures exactly the principal family. Under the former *marked* equality semantics this was exact: the marked reading equates the flank words on each side of the markers separately, which forces $x_2$ to absorb $s$ and excludes the sporadic solutions. Under the adopted $\approx$-semantics the sporadic solutions are genuine, and the binding would foreclose them — a *rank* commitment ($\mathrm{rank}(\gamma\rho_2) \ge |s|$), strictly stronger than a placement choice.

**The implementation does not bind here.** `unify_row`'s cross-surplus branch (`beg_handled = false`, `row.ml` ~2242) re-emits the residual equation and commits nothing: **deferral into closing**, with the same profile as the rotational case (d): if either variable is solved by other constraints, the substituted check (case (b), exact) decides; otherwise the guessing stage closes one variable upward — the least-material disjunct, which for $x_2 = [\,]$ is precisely the *outermost sporadic* solution — and the re-emitted equation decides the rest. Pinned by `test/einsum/test_row_self_reference.ml`: the store $\{[5]\cdot\langle\rho_1\rangle \approx \langle\rho_2\rangle\cdot[5,3],\ \langle\rho_2\rangle \approx [\,]\}$, satisfiable only sporadically ($\gamma\rho_1 = [3]$, $\gamma\rho_2 = [\,]$), succeeds in both constraint orders.

**The corrected rule (c-split), now official in the core:** defer; on substitution, re-check exactly; at closing, the upward close of either variable selects the corresponding extremal solution and the re-emitted equation validates it. The policy coordinate is *which* solution of the two-variable equation the closing selects (a finite choice — the $\le |s|$ sporadic splits — plus the family parameter $w$, itself pinned by whatever closes $\rho'$-content later); the deferral step itself is semantic.

*Soundness and the policy classification, assembled:* every rule above either (i) replaces a constraint by an equivalent set — semantic, $\mathrm{Sol}$ preserved exactly; (ii) moves constraints between $\Phi$ and $B$/$\sigma$ — denotation-preserving; or (iii) commits a representative beyond entailment — the placement coordinate of (b)/(c-nested) bindings, and the closing-stage resolution of the deferred equations of (d) and (c-split). Steps of kind (iii) are sound (the committed value satisfies the consumed constraint, so any solution of the new denotation satisfies the old), and their failures are policy rejections. $\square$

### A.1.4 Row inequality (Def. 4.3, cases i–iv) — all semantic

The ground meaning of $R_{\mathrm{res}} \sqsubseteq R_{\mathrm{op}}$ is Def. 2.3: flanks-fit plus pointwise refinement of the expansion, anchored at the outer edges.

- **Aligned-overlap dimension emissions**: positions covered by both sides' explicit flanks compare pointwise in any grounding (Def. 2.3(ii) at those positions, which by Lemma 2.4 depend on the grounding only through the compared dimensions): equivalent reformulation.
- **(i) operand open, result covers the flanks**: the remaining conditions say exactly that $\gamma\rho_{\mathrm{op}}$, expanded into the gap, is pointwise above the result's interior residue — i.e. residue $\sqsubseteq \gamma\rho_{\mathrm{op}}$ in the row order (Lemma 2.4 again: the conditions factor through the order). Recording the residue as a cap moves this into $\mathrm{constr}(B)$: denotation-preserving.
- **(ii) operand closed, result rank within bounds**: the operand's interior expansion positions are $1_\emptyset$; conditions of the form $\_ \sqsubseteq 1_\emptyset$ are valid and discarded.
- **(iii) result closed and shorter than the operand's known flanks**: flanks-fit requires $\mathrm{rank}(\gamma R_{\mathrm{res}}) \ge |l_{\mathrm{op}}| + |r_{\mathrm{op}}|$; the left side is fixed, the right side can only grow under substitution (explicit flanks extend inward only): unsatisfiable, semantic failure.
- **(iv) deficit rule**: for every solution $\gamma$, flanks-fit forces $\mathrm{rank}(\gamma\rho_{\mathrm{res}}) \ge k$, and $\gamma\rho_{\mathrm{res}}$ decomposes uniquely (marker-sensitively — the marker of $\gamma\rho_{\mathrm{res}}$ falls in the middle segment, since the template splices the fresh row variable at the marker) into $k_l$ leading dimensions, a middle, and $k_r$ trailing dimensions matching the template; instantiate the fresh variables by that decomposition. Conversely any solution of the grown configuration restricts. Exact, up to extension on fresh variables. The guarded failure (Def. 4.5) is semantic by Prop. 4.6's soundness direction: a positive cycle witnesses rank-unsatisfiability of entailed facts.

This completes Lemma 5.1. The policy steps are: **placements** at (b)/(c-nested) bindings, and **deferred-equation resolutions** at closing for (d) and (c-split). $\square$

---

## A.2 Lemma 5.2.2 (Rank conservativity) — rule by rule

**Statement.** Every non-$\mathsf{fail}$ rule maps a configuration whose denotation has a rank model (Def. 5.2.1) to one with a rank model agreeing on the old row variables, extended on fresh ones.

*Proof.* Let $r$ be a rank model of $\lfloor\widehat{C}_{\mathrm{old}}\rfloor$.

- **Dimension rules** (all of Def. 4.2): no row constraint is created, consumed, or rewritten at row positions; $\lfloor\widehat{C}\rfloor$ at row sort is unchanged (dimension substitution alters flank *content*, never flank *length*). Take $r$ unchanged.
- **Row equality (a)**: consumes a constraint contributing only variable-free rank facts (both ranks fixed; the non-fail case has them equal); emits dimension constraints only. $r$ unchanged.
- **Row equality (b)**: the consumed constraint contributes $\mathrm{rank}(\rho) = |w| - |l| - |r| = |m|$ (both directions of the equality). The binding contributes, via $\mathrm{eqns}(\sigma)$, exactly $\mathrm{rank}(\rho) = |m|$; substituting $\rho$'s ground value through other constraints shifts their facts by constants in a way that preserves $r$'s satisfaction (each occurrence $l'\cdot\langle\rho\rangle\cdot r'$ becomes a term whose explicit flank length grew by exactly $|m|$ — wait, by $|m_l| + |m_r| = |m| = r(\rho)$ — so every fact in which $\rho$'s host term participated keeps the same numeric content under $r$). $r$ unchanged.
- **Row equality (c-nested)**: the consumed constraint contributes $\mathrm{rank}(\rho_1) = \mathrm{rank}(\rho_2) + |m_l| + |m_r|$; the binding contributes the same fact; substitution as in (b). $r$ unchanged. (No fresh variables in the corrected rule.)
- **Row equality (c-split) and (d, rotational)**: deferral — the constraint stays in $\Phi$; denotation unchanged; $r$ unchanged. (Their *closing-stage* resolutions are policy commitments; conservativity for the closing pipeline is addressed in A.5, where the committed ground values either satisfy $r$'s constraints or fail finitely.)
- **Row inequality (i)**: moves pointwise conditions into a cap; the cap's rank fact (residue rank bounds $\mathrm{rank}(\rho_{\mathrm{op}})$ from below... in the order's direction: $\mathrm{rank}(\gamma\rho_{\mathrm{op}}) \le \mathrm{rank}(\mathrm{residue})$, an upper fact when the residue is closed, a difference fact when open) is the same fact the consumed constraint contributed. $r$ unchanged.
- **Row inequality (ii)/(iii)**: variable-free or failing; non-fail cases discard valid conditions. $r$ unchanged.
- **Row inequality (iv), deficit**: the consumed constraint's fact gives $r(\rho_{\mathrm{res}}) \ge k$ (flanks-fit against the operand's known flanks). Extend $r$ to the fresh row variable by $r(\rho') := r(\rho_{\mathrm{res}}) - k \ge 0$; the binding's fact ($\mathrm{rank}(\rho_{\mathrm{res}}) = \mathrm{rank}(\rho') + k$) holds by construction, and substitution preserves numeric content as in (b). Fresh dimension variables carry no rank facts.
- **Re-emission on binding**: each re-emitted bound was in $\mathrm{constr}(B)$, hence its fact was in $\lfloor\widehat{C}_{\mathrm{old}}\rfloor$ and $r$ satisfies it; instantiation at the bound value preserves numeric content as in (b).

In every case the new configuration's rank abstraction is satisfied by $r$ (extended). $\square$

*Remark.* The previously drafted (c-split) fresh-variable binding would **break** this lemma: its facts include $\mathrm{rank}(\rho_2) \ge |s|$, not implied by the consumed constraint's single difference fact, so a rank model with $r(\rho_2) < |s|$ — which exists precisely when the sporadic solutions are the live ones — could not be extended. This is the metatheoretic fingerprint of the same incompleteness that A.1.3 identifies semantically, and an independent reason the deferral is the right rule.

---

## A.3 Theorem 5.2(a) — the lineage bookkeeping bound

**Claim (the bracketed gap).** The number of row-inequality lineages ever live in a run is bounded by a function of $|\Phi_0|$, where a *lineage* is the identity of a row-inequality constraint tracked through residue formation, storage as a cap, re-emission, and substitution.

*Proof.* Define the lineage of a row inequality as follows: each row inequality in $\Phi_0$ starts one; the residue cap recorded by rule (i) *continues* the lineage of the consumed constraint, as does its re-emission upon the host variable's binding, as does the reprocessed constraint after a deficit step or any substitution. New row inequalities are *created* (not continued) at exactly two points: (1) re-emission of a stored adjacency — but an adjacency re-emission re-expresses the *same* stored inequality, continuing its lineage; (2) closing commitments re-emitting stored constraints — again continuations. Inspecting Defs. 4.2–4.3: **no rule emits a row inequality that does not continue the lineage of its consumed constraint** — dimension rules emit dimension constraints; equality rules emit dimension equalities and (deferred) row *equalities*; the deficit rule reprocesses the same constraint; rule (i) stores its own residue. Row *equalities* similarly: no rule creates a row equality except as a continuation ((c-split)/(d) deferrals re-emit the consumed equation; case (b)/(c-nested) consume theirs into bindings) — so row-equality lineages are also $\le$ their count in $\Phi_0$ plus the closing pipeline's per-variable commitments (one per variable, each variable closed at most once).

It remains to bound storage/re-emission multiplicity per lineage: a cap is stored on one variable and re-emitted when that variable binds — at most once, since a variable binds at most once and is then substituted out (the re-emitted continuation may be stored again, on a *different, currently unsolved* variable; each storage–re-emission round strictly advances the host along the substitution order, and hosts are drawn from the variables ever minted). This does not yet bound rounds by $|\Phi_0|$ — hosts can be fresh mints — but it does not need to: the potential argument of Thm. 5.2(a) charges each *deficit step* of a lineage against the strict decrease of $\psi(c) = \hat r(\text{current result variable})$, and $\psi$ never increases along any continuation (deficit decreases it by $\ge 1$; equality splices keep $\hat r$ equal or smaller; storage and re-emission do not change the result term). So per lineage the deficit steps are $\le \psi_0(c)$, and the lineage *count* (bounded above by $|\Phi_0|$ + one per closing commitment, itself bounded by the variable count at closing time) bounds total deficit steps. With deficits finite, mints are finite; with mints finite, bindings are finite; between bindings the lexicographic measure (row constraints, dimension constraints) strictly decreases as in the core's proof. $\square$

*(What made this routine is the observation that lineages are never **forked**: every rule emits at most one row-sort descendant of its consumed row-sort constraint. If a future rule forks — e.g. a backtracking branch materializing several sporadic solutions of A.1.3 — the bound becomes per-branch.)*

---

## A.4 Confluence across fair strategies — discharged by decision

The core's Thm. 5.6 caveat offered two routes: a Newman-style local-confluence argument, or fixing one strategy. **The paper fixes the strategy**: the implementation's — a per-round queue fixpoint over the constraint list with per-round accumulation reversal, wrapped in the staged pipeline of §10 (Stage 1 online unify/propagate; commitments confined to on-demand stages 2–7). With the strategy fixed, Thm. 5.6's uniqueness-up-to-renaming needs no cross-strategy argument, and the policy commitments (placements, deferred-equation resolutions, closing) are *deterministic functions of the input constraint order* — which is the precise content of Remark 2.15(v)'s "principality up to $\approx$, placement as order-sensitive policy."

Cross-strategy confluence (up to $\approx$ and modulo policy) remains true-by-expectation and **[open]**; nothing in the paper depends on it. The semantic-step core is confluent for satisfiability outright by Lemma 5.1 (each semantic step preserves $\mathrm{Sol}$ exactly, so any two runs preserve the same invariant); what varies across strategies is only which policy representative is reached — Remark 6.7(iii) in the core.

---

## A.5 Proposition 6.2 — the re-solve termination case split, strengthened

**Claim.** Every re-solve launched by a closing commitment terminates; consequently the closing interleave terminates, and on rank-breaking commitments it terminates in $\mathsf{fail}$ finitely.

*Proof.* A closing commitment binds one variable to a *ground* value. Partition the post-commitment denotation's rank abstraction into the **variable–variable fragment** (difference facts between still-open row variables) and the **absolute fragment** (facts with at least one ground side).

(1) *The variable–variable fragment retains a rank model.* Ground commitments add no variable-to-variable facts: substituting a ground value into a stored constraint leaves at most one variable in it, and re-emitted bounds are single-variable or ground. The pre-closing solved form's denotation had a rank model (Prop. 5.4's $\gamma_\uparrow$ induces one), and the var–var fragment is a subset of facts entailed then, extended only conservatively by subsequent semantic steps (Lemma A.2). 

(2) *Deficit steps in the re-solve are bounded.* The potential argument of Thm. 5.2(a) needs $\hat r$ only on **open-result lineages** ($\psi = 0$ on closed results by definition). Open-result lineages' facts live in the var–var fragment plus absolute lower bounds from ground operands; a model of the var–var fragment extends to the open-result potentials (absolute *upper* facts may be violated — but $\psi$ does not consult them; their violation surfaces as finite failures, not growth). So $\psi$ is well-defined and the deficit count is bounded as in A.3.

(3) *Everything else is finite.* With deficits bounded, mints and bindings are finite; between bindings the lexicographic measure decreases; ground-sided constraints are each processed in boundedly many steps (cases (a)/(b)/(ii)/(iii) and the dimension rules consume or fail; the one-shot closed-operand deficit grows a bounded amount once).

So the re-solve terminates unconditionally. If the commitment broke rank-satisfiability (necessarily through the absolute fragment, by (1)), the run cannot reach solved form — a solved form would yield, via $\gamma_\uparrow$, a solution and hence a rank model of the whole — so it fails, in the finite time just established. Soundness of the interleave is the core's existing argument. $\square$

*(This strengthens the core's Prop. 6.2: termination of the re-solves is **not** conditional on the committed system retaining a rank model — only success is. The Detection Lemma is not needed here because closing commitments are ground: the unbounded-growth mechanism of Example 4.5a requires variable–variable feeding, which ground commitments cannot create.)*

---

## A.6 Proposition 6.4(iii) — $\sqsubseteq^1$-greatestness, written out

**Statement (with the precise qualification).** Let a run succeed with solved form $\langle\emptyset; \sigma_\star; B_\star\rangle$ and let $\gamma_\uparrow$ be Prop. 5.4's solution. Let $\Phi_0^{\mathrm{rk}}$ be $\Phi_0$ plus the run's *deferred-equation resolutions* (the (c-split)/(d) closing commitments of A.1.3 — rank-level policy). Then for every $\gamma \in \mathrm{Sol}(\Phi_0^{\mathrm{rk}})$ and every $x \in V$:
$$\gamma(x) \sqsubseteq^{1} \gamma_\uparrow(x) \text{ at row sort}, \qquad \gamma(x) \sqsubseteq \gamma_\uparrow(x) \text{ at dimension sort.}$$
Placement commitments need no exclusion — $\sqsubseteq^1$ is marker-blind on its lower side, which is exactly why the one-shot order is the right comparison. Rank commitments do: a solution realizing a foreclosed sporadic/conjugate branch can have *smaller rank* than $\gamma_\uparrow$'s committed value at a variable (the pinning test's $\gamma\rho_1 = [3]$ against a family value $w\cdot[5,3]$), and no marker-blind order repairs a rank gap. On deferral-free runs $\Phi_0^{\mathrm{rk}} = \Phi_0$ and the statement is unconditional.

*Proof.* **Dimension sort.** If $x$ is unsolved in $\sigma_\star$: $\gamma_\uparrow(x) = 1_\emptyset$, the top. If solved, $\sigma_\star(x)$ is (by idempotence) a ground atom or an unsolved variable: in the first case the binding content is entailed (Lemma 5.1's semantic component — dimension bindings have no policy coordinate), so $\gamma(x)$ equals it, $= \gamma_\uparrow(x)$; in the second, $\gamma_\uparrow(x) = 1_\emptyset$ again.

**Row sort.** Let $x$ be a row variable. If unsolved: $\gamma_\uparrow(x) = [\,]\cdot\diamond\cdot[\,]$, and for any ground row $X$, $\mathrm{flat}(X)^\bullet \sqsubseteq [\,]\cdot\diamond\cdot[\,]$ — flanks fit vacuously (the marked side has none) and every position of the expansion is $1_\emptyset$. If solved with $T = \sigma_\star(x)$: the *flat content* of the binding is entailed for solutions of $\Phi_0^{\mathrm{rk}}$ — for (b)/(c-nested) bindings by Lemma 5.1's semantic component; for values that passed through deferred-equation commitments, by membership in $\Phi_0^{\mathrm{rk}}$ — so $\mathrm{flat}(\gamma(x)) = \mathrm{flat}(\hat\gamma(T))$ for an extension $\hat\gamma$ of $\gamma$ over the fresh variables (choosing the extension Lemma 5.1 provides). It remains to show $\mathrm{flat}(\hat\gamma(T))^\bullet \sqsubseteq \gamma_\uparrow(T)$, by induction on $T$:

- $T$ closed, $T = e_1 \cdots e_n$ at flat positions (each $e_i$ a ground atom, $1_\emptyset$, or a dimension variable): $\gamma_\uparrow(T)$ has the same rank with positions $\gamma_\uparrow(e_i)$; flanks fit (equal ranks, rigid lower side), and pointwise $\hat\gamma(e_i) \sqsubseteq \gamma_\uparrow(e_i)$ by the dimension-sort case. 
- $T = l\cdot\langle\rho'\rangle\cdot r$ open: $\gamma_\uparrow(T) = \gamma_\uparrow(l)\cdot\diamond\cdot\gamma_\uparrow(r)$ (the middle empty, $\gamma_\uparrow\rho' = [\,]\cdot\diamond\cdot[\,]$; if $\rho'$ is itself solved, recurse through $\sigma_\star$'s idempotent closure first). $\mathrm{flat}(\hat\gamma(T)) = \hat\gamma(l)\cdot\mathrm{flat}(\hat\gamma\rho')\cdot\hat\gamma(r)$ has rank $\ge |l| + |r|$: flanks fit. Pointwise against the expansion $\gamma_\uparrow(l)\cdot 1_\emptyset^{\,k}\cdot\gamma_\uparrow(r)$: flank positions by the dimension-sort case; middle positions sit under $1_\emptyset$. $\square$

---

## A.7 Remark 6.7(i) — order-robustness of closing, written out

**Statement.** At dimension sort, with cap stores transitively closed at commitment time (the fixpoint-before-commit discipline: every ground bound has been forwarded across the variable–variable adjacency to fixpoint) and no holes among the committed variables' caps, the leaf-downward closing of Def. 6.1 never fails on a store whose pre-closing denotation is satisfiable — in any commitment order consistent with the discipline.

*Proof.* By the discipline, for each leaf $v$ the cap set $\mathrm{caps}(v)$ at commitment time contains every ground bound reachable into $v$: in particular, $u \sqsubseteq v$ in the adjacency implies $\mathrm{caps}(u) \subseteq \mathrm{caps}(v)$ (each ground bound arriving at $u$ was forwarded), and a *later* commitment of $u$ adds nothing new to $v$'s caps, because $u$'s committed value is $J_u := \bigvee \mathrm{caps}(u)$, and the fact $J_u \sqsubseteq v$ is implied by $\mathrm{caps}(u) \subseteq \mathrm{caps}(v)$ ($J_u \sqsubseteq J_v \sqsubseteq \gamma(v)$ for any candidate). So the commitments are order-independent in value: each leaf gets $J_v$, regardless of sequence — closing under the discipline *is* the simultaneous assignment $\gamma_J(v) = J_v$ on leaves.

It remains to show $\gamma_J$ extends to a solution when one exists, i.e. no re-check fails. Let $\gamma$ be a solution of the pre-closing denotation. Constraint forms at dimension sort after solving (Def. 5.3): ground caps $c \sqsubseteq v$ — satisfied, $c \sqsubseteq J_v$ by the join; adjacencies $u \sqsubseteq v$ — satisfied, $J_u \sqsubseteq J_v$ by monotonicity of join under $\mathrm{caps}(u) \subseteq \mathrm{caps}(v)$; pins of either direction do not survive into solved form (Def. 5.3: no pending pins — DI-pin and DI-pin-top fire during solving). Mixed leaf–interior adjacencies: a leaf-commitment re-emission against an interior variable is a ground cap or a ground pin on the interior, handled by the (semantic) solving rules before step 3; step 3's upward closes then discharge against tops as in Prop. 6.2. Holes are excluded by hypothesis; with holes the commitment is a guess and Prop. 6.5's caveat applies instead. $\square$

*(Row sort: the same argument gives the flat content — cap-set inclusion and join monotonicity hold per Prop. 2.7 on ground parts — but the row join must additionally choose a placement and an extent beyond the caps (no-further-axes), both policy coordinates; route them to Remark 6.7(ii)(a). The implementation's discipline is architectural — Stage 1 only propagates; commitments are confined to stages 2–7 — per `docs/shape_inference.md` and the §10 audit.)*

---

## A.8 Remark 2.15(iv) — the inference-policy-vs-checking theorem

**Theorem.** Let $\Phi_{\mathrm{eq}}$ be a constraint set whose einsum-derived members are the equivalence constraints $\{T_i \approx R_i\}$ (tensor row $T_i$ against grounded-template row $R_i$, per Remark 2.15), and let $\Phi_{\sqsubseteq}$ be the same set with each einsum equivalence replaced by its checking sandwich — $\mathrm{res} \sqsubseteq^1 T_{\mathrm{res}}$ and $T_{\mathrm{opnd}} \sqsubseteq^1 \mathrm{opnd}$ in the core's formulation, i.e. the one-shot inequalities with the template in the middle. Then:

1. *(Per-constraint strengthening)* $X \approx Y$ implies $X \sqsubseteq^1 Y$ and $Y \sqsubseteq^1 X$. *Proof:* $\approx$ is mutual $\sqsubseteq^1$ — proved inline at the core's Remark 2.15(ii); the forward direction alone is Prop. 2.13's rigid-below-marked at equal rank with equal content. $\square$
2. *(System-level)* $\mathrm{Sol}(\Phi_{\mathrm{eq}}) \subseteq \mathrm{Sol}(\Phi_{\sqsubseteq})$. *Proof:* pointwise by (1) — each replaced constraint is weakened, the rest are shared. $\square$
3. *(Solver outputs are checking-correct)* If the staged solve of $\Phi_{\mathrm{eq}}$ succeeds, its committed total grounding $\gamma_{\mathrm{close}}$ satisfies $\Phi_{\sqsubseteq}$. *Proof:* $\gamma_{\mathrm{close}} \in \mathrm{Sol}(\Phi_{\mathrm{eq}})$ by Prop. 6.2 (soundness of closing over Lemma 5.1), then (2). $\square$
4. *(The strengthening is proper, and is the inference policy)* In general $\mathrm{Sol}(\Phi_{\mathrm{eq}}) \subsetneq \mathrm{Sol}(\Phi_{\sqsubseteq})$: the sandwich leaves the result row free below the template ($\mathrm{res}$ may be more material — rank-broader — than $T_{\mathrm{res}}$ requires) and operands free above; the equivalence pins ranks and transports dimension content bidirectionally. This is a commitment of the same nature as Def. 6.1's closing — selected for inference value (rank pinning is what determines hidden and parameter dimensions), justified against the checking semantics by (3): nothing the inference system accepts violates the sandwich. $\square$

*(What this does not claim: completeness relative to checking — a program could satisfy $\Phi_{\sqsubseteq}$ under some grounding yet be rejected under $\Phi_{\mathrm{eq}}$. That is the intended behavior — einsum equivalences are the stricter, inference-friendly reading — and the residual obligation about **which** programs separate the two readings belongs with the surface-language work of A.10.)*

---

## A.9 The Detection Lemma — structured reduction (the residual open gap)

**Statement (to prove).** Every infinite run inserts, at some finite stage, an edge closing a positive-weight cycle in the persistent rank graph $G$ of Def. 4.5.

**D1 (Infinite runs have infinitely many open–open deficit steps). [proved]** An infinite run has infinitely many binding events (between bindings the lexicographic measure of Thm. 5.2(a)'s proof strictly decreases through finitely many constraint-consumptions). Each variable binds at most once, so infinitely many variables are minted. Mints occur only at the deficit rule (row equalities are finitely many forever, by A.3's no-forking accounting, and (b)/(c-nested) mint nothing — the corrected (c-split) and (d) defer). Deficit steps against *closed* operands are bounded: each grows its lineage's result by a fixed amount toward a fixed rank, once (Def. 4.5(iii)'s one-shot remark), and lineages are finitely many (A.3). Hence infinitely many deficit steps fire with *open* operands — each recording a kind-(iii) positive edge and minting along its lineage's kind-(i) chain. $\square$

**D2 (Deficit accounting along the feeding relation). [sketch]** For lineages $A$ (result side) and $B$ (operand side at the moment of firing), say $A \leftarrow B$ ("$A$ fed by $B$"). A deficit on $A$ fires only when $A$'s operand flank is longer than $A$'s current result flanks; the operand flank grows only by (a) its $\Phi_0$-initial surplus, or (b) substitution of a binding made on $B$'s chain — each unit of $B$'s growth lengthens the operand side of any lineage reading $B$ by at most that unit (substitution preserves flank totals elsewhere; only the bound variable's expansion inserts material). Hence, as running totals: $\mathrm{deficit}(A) \le S_0(A) + \sum_{B : A \leftarrow B} \mathrm{growth}(B)$, and $\mathrm{growth}(B) = \mathrm{deficit}(B)$. Since lineages are finite in number and $S_0$ is finite, unbounded total deficit forces a cycle $A_1 \leftarrow A_2 \leftarrow \cdots \leftarrow A_n \leftarrow A_1$ in the (time-dependent) feeding relation around which deficit circulates unboundedly. **[The time-dependence is the unfinished part: feeding edges appear and move as substitutions re-target operands; the inequality must be stated against a fixed finite quotient — e.g. lineage pairs — with the growth bookkeeping per pair.]**

**D3 (Circulation transports into a $G$-cycle). [open — the residual gap]** Around one circulation of the feeding cycle, compose: the kind-(iii) cross edge recorded at each deficit (current result vertex $\to$ current operand vertex, weight $=$ the deficit), the kind-(i) chain edges along each lineage between the operand vertex used and the result vertex of the next firing (weights $=$ growth amounts), and the kind-(ii) zero-weight anchors tying mints to the operand vertices they were measured against. *Claim:* the composite is a closed directed walk in $G$ whose total weight equals the net circulation, which is positive for at least one circulation of an unboundedly-fed cycle; hence the guard fires at the edge completing it. The core's §5 hand-trace verifies this on the three-variable cycle (closed walk of weight 3 at the third constraint's first deficit). What is missing is the general bookkeeping: that the anchors always supply the return path (each mint's kind-(ii) edge points to the operand vertex, which lies on the *next* lineage's chain — direction must be checked against the sign discipline), and that the walk's weight telescopes to the circulation rather than cancelling. **[This is where the proof lives or dies; the sign discipline of Def. 4.5 (no edges for nonpositive entailments) is what makes the telescoping plausible — only positive growth is ever recorded — but also what could starve the return path in configurations where the feeding surplus sits in case-(i) residues, which record nothing.]**

*Status: reduced to D2's quotient bookkeeping and D3's transport argument. A counterexample hunt is also sensible: a configuration circulating deficit exclusively through case-(i) residues (which record no edges) would refute D3 as stated and force a fourth edge kind. The hunt should start from Example 4.5a variants with the growth laundered through caps.*

---

## A.10 Marker provenance and the surface discharge — statements

**Invariant (to prove).** In every configuration reachable from a surface-program-generated $\Phi_0$: the marker placement of every closed row and every solved value's split traces to a *declared origin* — an external (left-marked) row, a spec ellipsis position, or the slice operation's left-flank prepend (§10's audited origin list). Inductive obligations: each rule of Defs. 4.2–4.3 either preserves placements (substitution, zips), inherits them (the (b)/(c-nested) inherit-the-split policy — provenance-faithful by construction), or defers ((c-split), (d)); closing's joins and upward closes use the left-edge default, itself a declared-origin form.

**Surface discharge (conjecture, Cor. 5.5).** Provenance-respecting constraint sets never present a *placement-discriminating* store: no marked inequality whose satisfaction depends on which $\approx$-representative an equality binding committed. The probe (Remark 2.11's witness) requires an equality pinning a row's content together with a marked inequality carrying a *different* split of overlapping content — the §10 origin audit suggests einsum equalities and the pointwise sandwich cannot produce the mismatched pair, but the slice's left flank is the suspect to check first (it is the one surface construct that manufactures nontrivial leading flanks). **[open — needs the surface-language grammar formalized; deferred together with Lemma 7.7's elaboration audit.]**
