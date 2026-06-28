# Neural Networks Without Shape Boilerplate: An OCaml DSL Case Study

## Abstract

Modern neural network code is often cluttered with shape bookkeeping: explicit calls to reshape, unsqueeze, expand, transpose, reductions, and integer-axis arguments. OCANNL combines an embedded DSL that removes this boilerplate, with an end-to-end compiler with GPU backends. Users define tensor operations as OCaml functions using concise notation generalizing the Einstein summation convention. Concatenation is an expression over indices e.g. `i^j`, convolution is a contraction with affine operand addressing e.g. `stride*i + dilation*k`. A sequence of axes can be captured by a single variable, e.g. `..batch..` or `..activations..`, sandwiched between leading and trailing axes; with up to three such variables per shape. We call these *row variables*. The paper showcases three examples: tensor expressions core multi-head attention, rank-polymorphic 2D convolution; and tensor computation for Stochastic Gradient Descent with momentum and Nesterov-inspired correction. OCANNL performs broadcast-aware global bidirectional shape inference and derives loop-index maps for code generation. We formalize the inference problem for the core calculus (excluding affine and concatenation) and show properties of OCANNL's solver (proofs in the appendix). OCANNL's compiler inlines computations to avoid materializing intermediate tensors, and performs common subexpression elimination. This paper does not showcase benchmarks nor argue for performance, leaving that to follow-up work and the Q&A.

## Shapes, Syntax Extensions and Examples

OCANNL shapes have three kinds (sequences) of axes, from outermost: batch, output, input. The kind designation is by convention (not enforced, but facilitated). In specifications, the syntax resembles programming language types: `batch | input -> output`. For example: tensor multiplication `*` reduces the input axes of the operator tensor (matrix etc.) with output axes of the operand tensor (vector or matrix etc.), while broadcasting or treating pointwise the batch axes.

OCANNL introduces two syntaxes. Extension point `%op` creates *tensor expressions* (type `Tensor.t`), or OCaml functions returning tensor expressions which we call tensor *operations*. Extension point `%cd` creates tensor *computations* (assignments, type `Assignments.comp`). Unlike computations, tensor expressions are differentiable and support separate initialization. Both extensions support inline definitions of new tensors via OCaml's record syntax. For example: `{ w1 = kaiming normal1 () }` inside `%op` introduces tensor `w1` with initialization expression `kaiming normal1 ()`; `{ sgd_momentum }` inside `%cd` below introduces a (non-differentiable, no initialization) tensor `sgd_momentum`; `{ w_q }` inside `%op` below introduces a tensor with default initialization (e.g. centered uniform distribution).

At the heart of OCANNL are indexing specifications for expressing tensor computations. They generalize the Einstein summation convention: indices missing from the result are reduced over. The specification syntax uses `;` to separate arguments and `=>` to separate out the result. Ellipsis `...` in the specifications are expanded contextually as either `..batch..`, `..input..` or `..output..` row variables. Assignments in the `%cd` syntax specify the unary or binary arithmetic and the accumulation arithmetic explicitly. For the `%op` syntax, we have mixfix operators combining the arithmetic semantics, for example: `++` is identity with additive accumulation, `+*` is multiplication with additive accumulation. These operators also support *variable capture* for both dimensions and rows -- the variables from a trailing string list are introduced into scope earlier than they first appear (simiarly to inline definitions). The captured variables can be used for both shape constraints `Shape.set_dim`, `Shape.equal_dims` and converted to scalars via `dim` (for row variables, the value is the product of the dimensions of the axes).

```ocaml
let%op multi_head_attention ~num_heads ~d_k ~d_v () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ])
    /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights = softmax ~spec:" ... | t -> ..." () scores in
  { w_o } *
    (attn_weights +* v
       " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])
```
> **Figure 1. Core multi-head attention in OCANNL.** Axes: query position `s`, key/value position `t`, head `h`, key width `d`, and value width `e`. Batch rank, parameter shapes, score shape, contractions, and loop-index maps are inferred.

```ocaml
let%op conv2d ~label ?(kernel_size = 3) ?(stride = 1)
    ?(use_padding = true) ?out_channels () x =
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  Option.iter out_channels ~f:(Shape.set_dim oc);
  x +* { kernel }
       "... | stride*oh + kh, stride*ow + kw, ..ic..;
             kh, kw, ..ic.. -> ..oc..
        => ... | oh, ow, ..oc.."
       [ "kh"; "kw"; "oc" ]
  + { bias = 0. }
```
> **Figure 2. Rank-polymorphic 2D convolution.** The input is addressed at affine spatial positions `stride*oh+kh` and `stride*ow+kw`. The kernel axes `kh` and `kw`, together with the input-channel row `..ic..`, are contracted. The context row `...` and output-channel row `..oc..` are inferred and may have arbitrary rank.

```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0)
    ?(nesterov = false) p =
  [%cd
    { sgd_delta } =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      { sgd_momentum } =:
        (!.momentum *. sgd_momentum) + sgd_delta;
      if nesterov then sgd_delta =+ !.momentum *. sgd_momentum
      else sgd_delta =: sgd_momentum);
    p =- learning_rate * sgd_delta ~logic:"."]
```
> **Figure 3. Stochastic Gradient Descent with momentum.** Introduces intermediate state-tracking tensors in a computation. Optional Nesterov-inspired correction.

## Shapes and Projections: design choices and claims



## Contexts and explicit compilation

Unified, explicitly passed (immutable) context values. Empty root contexts determine device.

## Related work

## Conclusion

## Appendix: Shape and Projections inference: semantics and correctness

### 1. Dimensions

**Definition 1.1 (Dimensions).** Fix a set $\mathcal{B}$ of basis tags with a distinguished tag $\texttt{default} \in \mathcal{B}$. The set of dimensions is
$$D \;=\; \{1_\emptyset\} \;\cup\; \{\, n_b \mid n \ge 1,\; b \in \mathcal{B} \,\}.$$
Elements $n_b$ are *atoms*; $1_\emptyset$ is the *claim-free unit*. Adjoin a fresh element $\bot$ and set $D_\bot = D \cup \{\bot\}$.

**Definition 1.2 (Broadcast order).** On $D$: $d_1 \sqsubseteq d_2$ iff $d_2 = 1_\emptyset$ or $d_1 = d_2$. Extend to $D_\bot$ by $\bot \sqsubseteq d$ for all $d$.

**Proposition 1.3 (Lattice structure).** $(D_\bot, \sqsubseteq)$ is a bounded lattice of height $2$: top $1_\emptyset$, bottom $\bot$, and the atoms form an antichain, each covering $\bot$ and covered by $1_\emptyset$. Meets and joins:
$$d \wedge d = d,\quad d \wedge 1_\emptyset = d,\quad d_1 \wedge d_2 = \bot \ (d_1 \ne d_2 \text{ atoms});\qquad d \vee d = d,\quad d \vee \bot = d,\quad d_1 \vee d_2 = 1_\emptyset \ (d_1 \ne d_2 \text{ atoms}).$$

**Proposition 1.4 (Non-distributivity).** $D$ contains at least three atoms, e.g. $\{1_{\texttt{default}}, 2_{\texttt{default}}, 3_{\texttt{default}}\} \subseteq D$, so $D_\bot$ is not distributive, and not modular.

### 2. Rows

**Definition 2.1 (Rows).** A *row* is a triple $l \cdot \diamond \cdot r$ with $l, r \in D^{*}$ (the *leading* and *trailing flanks*). $\mathrm{rank}(l \cdot \diamond \cdot r) = |l| + |r|$. Equality of rows is componentwise equality of $(l, r)$ — in particular **marker-sensitive**: $[3]\cdot\diamond\cdot[4] \ne [3,4]\cdot\diamond\cdot[\,]$. (This is identity of the algebra's elements; it is *not* the satisfaction relation of equality *constraints*, which Def. 3.4 judges by the flat equivalence $\approx$ of Def. 2.9.)

**Definition 2.2 (Expansion).** For a row $R = l \cdot \diamond \cdot r$ and $n \ge \mathrm{rank}(R)$, the expansion $R{\uparrow}n \in D^{n}$ is the flat sequence $l \cdot (1_\emptyset)^{\,n - \mathrm{rank}(R)} \cdot r$.

**Definition 2.3 (Row order).** $l_2 \cdot \diamond \cdot r_2 \;\sqsubseteq\; l_1 \cdot \diamond \cdot r_1$ iff
(i) $|l_1| \le |l_2|$ and $|r_1| \le |r_2|$ (*flanks fit*; note this implies $\mathrm{rank}_2 \ge \mathrm{rank}_1$), and
(ii) $(l_2 \cdot r_2)[i] \sqsubseteq (l_1\cdot\diamond\cdot r_1){\uparrow}\mathrm{rank}_2\,[i]$ for all $i$ (*pointwise refinement of the expansion*).

**Lemma 2.4 (Expansion monotonicity).** If $R_2 \sqsubseteq R_1$ then for every $n \ge \mathrm{rank}(R_2)$: $R_2{\uparrow}n\,[i] \sqsubseteq R_1{\uparrow}n\,[i]$ for all $i$.

**Proposition 2.5 (Partial order).** $\sqsubseteq$ is a partial order on rows.

**Proposition 2.6 (The empty row is the greatest element).** For every row $R$: $R \sqsubseteq [\,]\cdot\diamond\cdot[\,]$, and nothing else is above all rows.

**Proposition 2.7 (Rows form a join-semilattice with top).** Any two rows $R_1, R_2$ have a least upper bound:
$$R_1 \vee R_2 \;=\; l_\vee \cdot \diamond \cdot r_\vee, \qquad |l_\vee| = \min(|l_1|,|l_2|),\ \ l_\vee[i] = l_1[i] \vee l_2[i],$$
and symmetrically for $r_\vee$ aligned from the back (joins taken in $D$; note two distinct atoms join to $1_\emptyset$, so $l_\vee \in D^{*}$).

**Remark 2.8 (Rank-polymorphism lives in the order).** Definition 2.3 relates rows of different ranks directly; no quantifier appears anywhere in this section.

**Definition 2.9 (Flat equivalence).** $R \approx R'$ iff $\mathrm{flat}(R) = \mathrm{flat}(R')$ as sequences, where $\mathrm{flat}(l\cdot\diamond\cdot r) = l \cdot r$ — equivalently, $\mathrm{rank}(R) = \mathrm{rank}(R')$ and $R{\uparrow}\mathrm{rank}(R) = R'{\uparrow}\mathrm{rank}(R')$, since a row's expansion at its own rank is its flattening (Def. 2.2 with no padding). $\approx$ is the marker-erasing equivalence; it is what the implementation's closed-row "equality" computes (§10).

**Proposition 2.10 ($\approx$ versus the order).**
(i) The equivalence induced by $\sqsubseteq$ (mutual inequality) is the **identity** — this is just antisymmetry (Prop. 2.5). So $\approx$ is not derivable from the order.
(ii) Stronger: any two *distinct* $\approx$-related rows are $\sqsubseteq$-**incomparable**.
(iii) $\approx$ is not a $\sqsubseteq$-congruence; indeed no nontrivial equivalence is compatible with the order.
(iv) $\approx$ is the kernel of expansion at own rank; everything else the order consults — flank-fit, and expansions at strictly higher ranks (broadcasting) — is exactly where $\approx$-related rows diverge.

The implementation does not use the observational equivalence of rows-as-operands: its equality sites use $\approx$ (observational equivalence is finer), its order sites the marked order (observational equivalence is looser because it absorbs explicit inner-edge $1_\emptyset$ entries into the middle).

**Remark 2.11 (Equality-as-$\approx$ is underdetermined; the marker placement is a policy choice).** Reading the ground meaning of closed-row equality as $\approx$ (forced by practice: an einsum spec row must equate with a literal shape whose marker sits at the front — §10) makes row-equality constraints underdetermined: in $l_1\cdot\langle\rho\rangle\cdot r_1 \approx C$, the flat content of $\gamma\rho$ is forced (the middle of $\mathrm{flat}(C)$) but its marker placement is free, and by Prop. 2.10(ii) the candidates are pairwise $\sqsubseteq$-incomparable — a later *inequality* on $\rho$ can distinguish them. Witness: $\Phi = \{\,\langle\rho\rangle \approx [\,]\cdot\diamond\cdot[3,5],\ \ [3]\cdot\diamond\cdot[9,5] \sqsubseteq \langle\rho\rangle\,\}$. Under $\approx$, $\gamma\rho = [3]\cdot\diamond\cdot[5]$ satisfies both ($[3,9,5]$ vs $[3,1_\emptyset,5]$). The implementation is *$\approx$-checking but marked-committing*: it accepts the flat alignment, then commits the closed side's structural split as the value's marker placement — on this $\Phi$ it fails. Def. 3.4 now **adopts** the $\approx$ reading, so that failure is officially a *policy rejection* (Lemma 5.1's policy steps), not semantic unsatisfiability: closed-row equality joins closing (Remark 6.5) as a deliberately non-principal site.

**Definition 2.12 (Two-sorted ground rows — proposal).** Adjoin to the marked rows of Def. 2.1 a second sort of *rigid* rows: $F^\bullet$ with $F \in D^n$, a flat sequence with **no marker**; $\mathrm{rank}(F^\bullet) = n$, and the expansion $F^\bullet{\uparrow}m$ is defined only at $m = n$, where it is $F$. Extend $\sqsubseteq$:
(a) marked–marked: Def. 2.3 unchanged;
(b) $F_2^\bullet \sqsubseteq F_1^\bullet$ iff equal rank and pointwise refinement;
(c) $F^\bullet \sqsubseteq R$ ($R$ marked) iff $\mathrm{rank}(F) \ge \mathrm{rank}(R)$ and $F[i] \sqsubseteq R{\uparrow}\mathrm{rank}(F)\,[i]$ pointwise — a rigid result may refine a broadcastable operand;
(d) $R \sqsubseteq F^\bullet$: **never** — nothing broadcastable sits below a rigid row.

**Proposition 2.13 (The two-sorted order).** The extended relation is a partial order; the empty marked row remains the unique top; and on the rigid sort, flat equality *is* equality — trivially a congruence, since no context consults a marker that does not exist. Admitting even rank-equal $R \sqsubseteq F^\bullet$ would break transitivity — $[5]\cdot\diamond\cdot[3] \sqsubseteq [\,]\cdot\diamond\cdot[3] \sqsubseteq [3]^\bullet$ would demand $[5]\cdot\diamond\cdot[3] \sqsubseteq [3]^\bullet$, a rank mismatch against a rigid row. The desired relation exists in the system as the derived *one-shot* relation — "$R \sqsubseteq F^\bullet$ at equal rank, pointwise" is literally $R \sqsubseteq^{1} F^\bullet$ of Remark 2.15.

**Remark 2.14 (What the rigid sort buys, and what it costs).** TODO: re-introduce once we know what's needed by later sections.

**Remark 2.15 (Einsum constraints: kind-indexed ellipsis, checking vs inference, and the one-shot order).** TODO: re-introduce once we know what's needed by later sections.


### 3. Terms and substitutions

**Definition 3.1 (Terms).** Dimension terms $t ::= \alpha \mid n_b \mid 1_\emptyset$. Row terms $R ::= l \cdot \diamond \cdot r \mid l \cdot \langle\rho\rangle \cdot r$ with $l, r$ sequences of dimension terms; we identify a bare row variable $\rho$ with $[\,]\cdot\langle\rho\rangle\cdot[\,]$. A term is *ground* if variable-free. Each row term contains **at most one** row variable, at the marker — an invariant preserved by everything below.

**Definition 3.2 (Substitution).** A substitution $\sigma$ is a finite, sort-respecting map from variables to terms with $\mathrm{dom}(\sigma) \cap \mathrm{vars}(\mathrm{ran}(\sigma)) = \emptyset$ (idempotency). Application is structural except at the marker: if $\sigma(\rho) = l' \cdot \langle\rho'\rangle \cdot r'$ (or closed, $l'\cdot\diamond\cdot r'$), then
$$\sigma(l \cdot \langle\rho\rangle \cdot r) \;=\; (\sigma l \cdot l') \cdot \langle\rho'\rangle \cdot (r' \cdot \sigma r) \quad\big(\text{resp. } (\sigma l \cdot l') \cdot \diamond \cdot (r' \cdot \sigma r)\big).$$

**Lemma 3.3 (Composition).** $(\sigma \circ \tau)(X) = \sigma(\tau(X))$ for all terms $X$, where $\sigma \circ \tau$ is the usual composite substitution.

**Definition 3.4 (Semantics).** A *ground substitution* $\gamma$ is total on a fixed countable variable universe and maps into ground dimensions/rows. For a constraint set $\Phi$ over atomic constraints $t_1 = t_2 \mid t_1 \sqsubseteq t_2 \mid R_1 \approx R_2 \mid R_1 \sqsubseteq R_2$:
$$\mathrm{Sol}(\Phi) = \{\gamma \text{ ground} \mid \gamma(\phi) \text{ holds for every } \phi \in \Phi\},$$
where ground *dimension* equalities are identity, ground *row* equalities are judged by the flat equivalence $\approx$ of Def. 2.9, equivalently, identity of the rigidifications $\mathrm{flat}(\cdot)^\bullet$ in the two-sorted algebra of Def. 2.12, equivalently mutual $\sqsubseteq^{1}$; the three formulations coincide. A substitution $\sigma$ *models* $\Phi$, $\sigma \models \Phi$, iff $\gamma \circ \sigma \in \mathrm{Sol}(\Phi)$ for every ground $\gamma$, equivalently, $\sigma(\Phi)$ is valid under universal quantification of its free variables. The *substitution order*: $\sigma_1 \le \sigma_2$ iff $\exists u.\ u \circ \sigma_1 = \sigma_2$.

**Example 3.5 (No principal model exists in general).** Let $\Phi = \{3_b \sqsubseteq \alpha\}$. The identity does not model $\Phi$ (ground $\alpha \mapsto 5_b$ falsifies it). Any model must map $\alpha$ to a term all of whose groundings lie above $3_b$, i.e. to $3_b$ or $1_\emptyset$ (a variable target fails as the identity did). The two models $\sigma_1 = [\alpha \mapsto 3_b]$ and $\sigma_2 = [\alpha \mapsto 1_\emptyset]$ are $\le$-incomparable: a witness $u$ for either direction would have to send a ground dimension to a different ground dimension, which substitutions cannot do. Hence $\Phi$ has models but **no $\le$-least model**.

*Consequence.* The solver's answer is a pair — a substitution *and a residual bound store* — and the correct theorem is about solution-set representation and most-generality (Theorem 5.6 below).

### 4. The solver: configurations and rules

**Definition 4.1 (Configuration).** A configuration is $\langle \Phi;\ \sigma;\ B \rangle$ — unsolved constraints, accumulated (idempotent) bindings, and a *bound store*. $B$ assigns to each unsolved variable: for a dimension variable $\alpha$, at most one atom lower bound $\mathrm{lb}(\alpha) \in \{n_b\}$ plus a set of deferred variable–variable inequalities ("adjacencies"); for a row variable $\rho$, a set of row-term lower bounds ("caps", from inequality residues) plus adjacencies. There is a distinguished failure configuration $\mathsf{fail}$.

Binding a variable ($\sigma := [x \mapsto T] \circ \sigma$) **re-emits** into $\Phi$ every bound and adjacency stored for $x$, instantiated at $T$, and substitutes $T$ for $x$ throughout $\Phi$ and $B$. This single discipline replaces ad-hoc propagation.

At dimension sort, the stored $\mathrm{glb}(\alpha)$ is understood as the join of all ground lower bounds known for $\alpha$, including those entailed through directed adjacencies. Operationally, when $d \sqsubseteq \alpha$ is recorded, the solver enqueues $d \sqsubseteq \beta$ for every stored edge $\alpha \sqsubseteq \beta$; when a new edge $\alpha \sqsubseteq \beta$ is recorded, it enqueues every stored cap $d \sqsubseteq \alpha$ as $d \sqsubseteq \beta$. Fair processing iterates these emissions to fixpoint. This is not a closing policy: it is the transitivity step $d \sqsubseteq \alpha \sqsubseteq \beta \Rightarrow d \sqsubseteq \beta$, and DI-cap handles any collapse produced downstream.

**Definition 4.2 (Dimension rules).**

| | Constraint | Action |
|---|---|---|
| DE-refl | $t = t$ | discard |
| DE-clash | $n_b = m_c$, $(n,b)\ne(m,c)$ (incl. atom vs $1_\emptyset$) | $\mathsf{fail}$ |
| DE-bind | $\alpha = t$, $t \ne \alpha$ | bind $\alpha \mapsto t$ |
| DI-top | $t \sqsubseteq 1_\emptyset$ | discard |
| DI-refl | $t \sqsubseteq t$ | discard |
| DI-ground | $d \sqsubseteq d'$ ground | check Def. 1.2; else $\mathsf{fail}$ |
| DI-pin | $\alpha \sqsubseteq d$, $d$ an atom | replace by $\alpha = d$ |
| DI-pin-top | $1_\emptyset \sqsubseteq \alpha$ | replace by $\alpha = 1_\emptyset$ |
| DI-cap | $d \sqsubseteq \alpha$, $d$ an atom | if $\mathrm{lb}(\alpha)$ unset: record and forward along stored outgoing adjacencies; if $= d$: discard; if $= d' \ne d$: replace by $\alpha = 1_\emptyset$ |
| DI-adj | $\alpha \sqsubseteq \beta$, $\alpha \ne \beta$ | defer (record adjacency on both) and forward $\alpha$'s stored cap, if any, to $\beta$ |

*Justifications (each rule preserves $\mathrm{Sol}$):* DI-pin: in $D$, the only element $\sqsubseteq$ an atom $d$ is $d$ itself. DI-pin-top: the only element above the top is the top. DI-cap collapse: any $\gamma(\alpha)$ above two distinct atoms must be $1_\emptyset$ (Prop. 1.3). DI-cap/DI-adj forwarding emits only transitive consequences of already-stored constraints. DI-pin-top is the case the blog's narrative omits.

**Definition 4.3 (Row rules).** Write both sides aligned at the **outer edges**:
leading flanks from the front, trailing flanks from the back.

For equality, first emit dimension equalities on the overlapping aligned flank positions. Let
$m_l, m_r$ be the surpluses: the unmatched inner segments of the longer leading and trailing
flanks. "Defer into the closing policy" means postpone a constraint till a later stage of the solving process.

*Equality $R_1 \approx R_2$.*

| | Situation | Action |
|---|---|---|
| RE-closed | both sides closed | $\mathsf{fail}$ iff the ranks differ; otherwise zip the flattened rows, emitting $\mathrm{flat}(R_1)[i] = \mathrm{flat}(R_2)[i]$ |
| RE-open-closed | one side open with $\rho$, the other closed | $\mathsf{fail}$ iff the open side's total explicit flank length exceeds the closed side's rank; otherwise bind $\rho$ to the uncovered middle of $\mathrm{flat}(R_{\mathrm{closed}})$, placed by the inherit-the-split policy |
| RE-nested | both sides open, $\rho_1 \ne \rho_2$, and both surpluses are on $\rho_2$'s side | bind $\rho_1 \mapsto m_l \cdot \langle\rho_2\rangle \cdot m_r$ |
| RE-cross | both sides open, $\rho_1 \ne \rho_2$, with split surpluses | defer the residual word equation $s \cdot x_1 = x_2 \cdot t$ into the closing policy |
| RE-same-empty | both sides open with the same $\rho$ and no surplus | discard |
| RE-same-occurs | both sides open with the same $\rho$ and unequal known flank totals | $\mathsf{fail}$ |
| RE-same-rot | both sides open with the same $\rho$ and equal totals but shifted splits | defer the rotational equation $x \cdot t = s \cdot x$ into the closing policy |

*Justifications and policy choices.* RE-closed is exactly Def. 3.4's $\approx$-semantics: markers
are not consulted. In RE-open-closed, substitution extends the open side's flanks inward only, so no grounding can shorten them. The binding
$\rho \mapsto m_l \cdot \diamond \cdot m_r$ with $m_l \cdot m_r = m$ additionally chooses a
placement that $\approx$ leaves free (Remark 2.11): it commits to the closed side's declared split
where that split falls inside $m$, and to the left edge otherwise. This is a policy step in the family of Def. 6.1. *(The implementation additionally
rejects when the open side's trailing flank overflows the closed side's; conservative under $\approx$.)*

RE-nested is entailed by cancelling common outer flanks of the flat words; only placement is policy,
as above. RE-cross is the two-variable word equation case: its solutions are the principal family
$x_2 = s\cdot w$, $x_1 = w\cdot t$ **plus** at most $|s|$ *sporadic* cross-overlap solutions
($x_2$ a proper prefix of $s$, the rest forced). Consider an alternative: fresh-variable binding
$\rho_2 \mapsto s\cdot\langle\rho'\rangle$, $\rho_1 \mapsto \langle\rho'\rangle\cdot t$. It captures
exactly the principal family: exact under the former marked semantics, but under $\approx$ it
forecloses the sporadic solutions, a *rank* commitment stronger than a placement choice (and it
breaks rank conservativity of Lemma 5.2.2). Deferring keeps the equation in flight; substituting
either variable makes the check exact, and the closing stage's upward close selects the corresponding
extremal solution (for the variable carrying $x_2$, the outermost *sporadic* one), then the
re-emitted equation validates it.

RE-same-occurs fails because no finite row satisfies
$\rho \approx m_l \cdot \langle\rho\rangle \cdot m_r$ with $m_l \cdot m_r$ nonempty. RE-same-rot is different. Its residue
$x \cdot t = s \cdot x$ is satisfiable under Def. 3.4's $\approx$-semantics exactly for conjugate
$s, t$ with cyclically periodic $x$ (Lyndon–Schützenberger), a solution family outside the solver's
constraint language. Deferring lets other constraints solve $\rho$ and make the substituted
closed-closed flat check exact. Otherwise, guessing closes $\rho$ upward — the least-material
disjunct, $x = [\,]$ — after which the equation requires $s = t$. This accepts exactly the
$\delta = 0$ conjugacy class: a policy commitment in the family of Def. 6.1 and since
$\approx$ is the adopted semantics, a genuine incompleteness on nontrivial conjugate instances
(periodic $x \ne [\,]$), accepted deliberately.

For inequality, first emit dimension inequalities $t_{\mathrm{res}} \sqsubseteq t_{\mathrm{op}}$
on aligned flank overlaps.

*Inequality $R_{\mathrm{res}} \sqsubseteq R_{\mathrm{op}}$.*

| | Situation | Action |
|---|---|---|
| RI-cap | operand open, and result supplies at least the operand's flanks | record the result's interior residue as a **cap** on $\rho_{\mathrm{op}}$; the residue is a row term, open if the result is open |
| RI-closed-op | operand closed, and result rank is at least the operand flanks | discard the operand's interior expansion; it is $1_\emptyset$, and the corresponding result positions are unconstrained |
| RI-short-closed | result closed and shorter than the operand's known flanks | $\mathsf{fail}$ |
| RI-deficit | result open and shorter than the operand's known flanks, with deficit $k > 0$ | bind $\rho_{\mathrm{res}} \mapsto \alpha_1 \cdots \alpha_{k_l} \cdot \langle\rho'\rangle \cdot \beta_1 \cdots \beta_{k_r}$ with fresh dimension variables and a fresh row variable, sized to clear the deficit; then reprocess |

RI-short-closed is a genuine rank mismatch. RI-deficit is subject to the rank-cycle check of
Def. 4.5.

**Definition 4.4 (Fairness).** A run processes constraints until $\Phi = \emptyset$ or $\mathsf{fail}$; re-emitted constraints return to $\Phi$. Any fair strategy is allowed.

**Example 4.5a (Divergence without a cycle check).** Let $\Phi_0 = \{\, \rho_1 \sqsubseteq [a]\cdot\langle\rho_2\rangle,\ \ \rho_2 \sqsubseteq [b]\cdot\langle\rho_1\rangle \,\}$ ($a, b$ ground atoms). Semantically: the first constraint forces $\mathrm{rank}(\gamma\rho_1) \ge 1 + \mathrm{rank}(\gamma\rho_2)$ (flanks-fit in Def. 2.3), the second forces $\mathrm{rank}(\gamma\rho_2) \ge 1 + \mathrm{rank}(\gamma\rho_1)$; hence $\mathrm{Sol}(\Phi_0) = \emptyset$. Operationally, RI-deficit fires on $\rho_1$, growing it by one axis; substitution lengthens the *operand* side of the second constraint, where RI-deficit grows $\rho_2$; substitution lengthens the operand side of the first constraint's residue, and so on forever. The rules of Def. 4.2–4.3 alone do not terminate. The self-cyclic instance $\rho \sqsubseteq [a]\cdot\langle\rho\rangle$ (obtainable by an einsum equality merging two row variables) diverges the same way. Developing this formalization uncovered an implementation bug: due to other mechanisms, the minimal *live* divergence was the three-variable cycle $\{\rho_1 \sqsubseteq \langle\rho_2\rangle{\cdot}[a],\ \rho_2 \sqsubseteq \langle\rho_3\rangle{\cdot}[b],\ \rho_3 \sqsubseteq \langle\rho_1\rangle{\cdot}[c]\}$, which ran unboundedly until killed (now rejected by the Def. 4.5 check).

**Definition 4.5 (Rank-fact graph and cycle check).** Maintain beside the configuration a directed graph $G$ on row variables — **persistent**: edges are only ever added, never retracted, in particular not when a variable is solved and substituted away. An edge $\rho \xrightarrow{\,k\,} \rho'$ with weight $k \ge 0$ records the entailed fact $\mathrm{rank}(\gamma\rho) \ge \mathrm{rank}(\gamma\rho') + k$ (for every solution $\gamma$ of the current configuration). Edges are recorded at three points:

(i) *binding:* solving $\rho \mapsto l \cdot \langle\rho'\rangle \cdot r$ records $\rho \xrightarrow{\,|l|+|r|\,} \rho'$ (the entailed fact is an equality; one direction suffices for the check; this covers RE-nested bindings and open row bindings produced by RI-deficit);
(ii) *equal flanks:* processing $R_{\mathrm{res}} \sqsubseteq R_{\mathrm{op}}$ with both sides open and equal known flank lengths records $\rho_{\mathrm{res}} \xrightarrow{\,0\,} \rho_{\mathrm{op}}$;
(iii) *deficit:* RI-deficit with deficit $k > 0$ and open operand records $\rho_{\mathrm{res}} \xrightarrow{\,k\,} \rho_{\mathrm{op}}$ *before* growing (a closed operand needs no edge: the rank bound is absolute and the growth is one-shot).

Every insertion is **guarded**: if the new edge closes a directed cycle of total weight $> 0$, then $\mathsf{fail}$.

*Soundness of failing:* summing the facts around a positive cycle gives $\mathrm{rank}(\gamma\rho) \ge \mathrm{rank}(\gamma\rho) + w$ with $w > 0$, so $\mathrm{Sol} = \emptyset$. *Soundness of persistence:* each edge is entailed by constraints present when it was recorded, and the solver never retracts a constraint — Lemma 5.1 only moves and re-expresses them — so entailed facts stay entailed, even when the variables involved have long been substituted away. Zero-weight cycles are legal: they assert rank *equality* along the cycle (and could be collapsed to row equalities, generalizing the one-step row antisymmetry the implementation already performs — cf. §10).

*Sign discipline.* In RI-cap — operand open, result's known flanks in *surplus* by $s > 0$ — the constraint entails only $\mathrm{rank}(\gamma\rho_{\mathrm{res}}) \ge \mathrm{rank}(\gamma\rho_{\mathrm{op}}) - s$: a *nonpositive* lower bound. Recording it with weight $0$ (or $s$) is **unsound**: it manufactures cycles that no solution violates. A first implementation attempt did exactly this and rejected legitimate programs (the SGD self-reference idiom among them); the fixed implementation records nothing there. Recording a genuinely negative weight would be sound but is unnecessary for the check's purpose.

**Proposition 4.6 (The check is exact for the recorded facts).** At any point in a run, the recorded fact set $\{\mathrm{rank}(\rho) \ge \mathrm{rank}(\rho') + k\}$ is satisfiable over $\mathbb{N}$ iff $G$ has no positive-weight cycle. Hence the guard fails exactly when the facts recorded so far are jointly unsatisfiable.


*(Standard difference-constraints/Bellman–Ford duality — worth stating because it delimits what the check decides: unsatisfiability of the **recorded** facts, a sound under-approximation of rank-unsatisfiability of the configuration. Whether enough facts are always recorded in time is exactly the Detection Lemma of Thm. 5.2(b).)*

### 5. Metatheory of solving

**Lemma 5.1 (Solution preservation, with a policy coordinate).** Write $\widehat{C}$ for the constraint set $\Phi \cup \mathrm{eqns}(\sigma) \cup \mathrm{constr}(B)$ denoted by a configuration ($\mathrm{eqns}(\sigma) = \{x \approx \sigma(x)\}$; $\mathrm{constr}(B)$ the stored bounds/adjacencies as constraints), and consider $\mathrm{Sol}(\widehat{C})$ under Def. 3.4. The rules of Defs. 4.2–4.3 divide:

*Semantic steps* — all dimension rules, all row-inequality rules (RI-cap, RI-closed-op, RI-short-closed, RI-deficit), and the checking content of the equality rules (the flat zip of RE-closed, the outer-edge dimension equalities, the rank-overflow failure of RE-open-closed, the discard of RE-same-empty, the occurs failure of RE-same-occurs, and the entailed flat content of RE-open-closed and RE-nested bindings) — preserve $\mathrm{Sol}(\widehat{C})$ exactly, up to extension of $\gamma$ to freshly introduced variables (RI-deficit): every solution of the old configuration extends to one of the new, and every solution of the new restricts to one of the old. $\mathsf{fail}$ steps of this kind are taken only when $\mathrm{Sol}(\widehat{C}) = \emptyset$.

*Policy steps* — the **placement** component of the row-variable bindings in RE-open-closed and RE-nested, and the **deferred-equation resolutions** at closing for RE-cross and RE-same-rot (both word-equation residues resolved by the upward close) — *refine* $\mathrm{Sol}$: the bound variable's flat content is entailed, but the committed split selects one of the pairwise $\sqsubseteq$-incomparable representatives of Prop. 2.10(ii), and substituting the committed value into stored *marked inequalities* may strengthen them. Every solution of the new configuration restricts to a solution of the old; conversely a solution of the old survives into the new after re-placing the bound variables' markers, which is possible iff no marked inequality of $\widehat{C}$ discriminates the placement — Remark 2.11's witness is the counterexample, and Remark 2.15(v)'s order-dependence is its operational trace. A $\mathsf{fail}$ arising from a substituted placement, where another representative would have passed, is therefore a *policy rejection*, not semantic unsatisfiability.

**Definition 5.2.1 (Rank abstraction).** For a configuration with constraint denotation $\widehat{C}$, its *rank abstraction* $\lfloor\widehat{C}\rfloor$ is the finite set of rank facts read off one constraint at a time: a row constraint relating $l\cdot\langle\rho\rangle\cdot r$ to $l'\cdot\langle\rho'\rangle\cdot r'$ (as result/operand of $\sqsubseteq$, resp. as an equality) contributes $\mathrm{rank}(\rho) \ge \mathrm{rank}(\rho') + (|l'|+|r'|) - (|l|+|r|)$ (both directions for an equality); open-vs-closed pairs contribute the corresponding absolute bounds on $\mathrm{rank}(\rho)$. A *rank model* is an assignment $r : \mathrm{RowVars} \to \mathbb{N}$ satisfying $\lfloor\widehat{C}\rfloor$. Every $\gamma \in \mathrm{Sol}(\widehat{C})$ induces a rank model $\rho \mapsto \mathrm{rank}(\gamma\rho)$ (flanks-fit in Def. 2.3); the converse fails — the abstraction forgets dimensions and bases.

**Lemma 5.2.2 (Rank conservativity).** Every non-$\mathsf{fail}$ rule maps a configuration with a rank model to one with a rank model agreeing on the old variables (extended on fresh ones). RI-deficit extends by $r(\rho') := r(\rho_{\mathrm{res}}) - k \ge 0$, justified by the triggering constraint's own rank fact; everything else preserves the model verbatim. *(A.2's remark: the previously drafted RE-cross fresh-variable binding would break this lemma — its $\mathrm{rank}(\rho_2) \ge |s|$ commitment is not implied by the consumed fact — an independent reason for RE-cross's correction to deferral.)* $\square$

**Theorem 5.2 (Termination).** Let $\Phi_0$ be finite.

**(a)** If $\Phi_0$ has a rank model — in particular whenever $\mathrm{Sol}(\Phi_0) \ne \emptyset$ — then every fair run terminates, with or without the cycle check; moreover the check never fires on such inputs (its recorded facts are entailed, hence satisfied by the rank model, hence positive-cycle-free by Prop. 4.6), so guarding is semantically free.

**(b)** If $\Phi_0$ has no rank model, then $\mathrm{Sol}(\Phi_0) = \emptyset$ and no run reaches solved form (Prop. 5.4 plus Lemma 5.1 would otherwise produce a solution, hence a rank model); every run either fails or diverges, and the cycle check is what must convert divergence into failure. A small gap remains in our proof that it always does. The proof is build around:

**Detection Lemma.** Every infinite run inserts, at some finite stage, an edge closing a positive-weight cycle in $G$. (Hence, with the guard of Def. 4.5, infinite runs do not exist, and rank-unsatisfiable inputs fail in finite time.)

**Definition 5.3 (Solved form).** A final configuration $\langle \emptyset; \sigma_\star; B_\star \rangle$ is in *solved form*: $\sigma_\star$ idempotent; every variable in $B_\star$ unsolved; each dimension variable carries at most one atom lower bound; all stored row caps and adjacencies are between unsolved variables and contain no pending pins (else a rule would fire).

**Proposition 5.4 (The residual store is always satisfiable; $\gamma_\uparrow$ is its greatest solution). [proved]** Define $\gamma_\uparrow$ on the unsolved variables: $\gamma_\uparrow(\alpha) = 1_\emptyset$, $\gamma_\uparrow(\rho) = [\,]\cdot\diamond\cdot[\,]$, extended through $\sigma_\star$ on solved variables ($\gamma_\uparrow(x) = \gamma_\uparrow(\sigma_\star(x))$). Then (i) $\gamma_\uparrow \in \mathrm{Sol}(\mathrm{constr}(B_\star) \cup \mathrm{eqns}(\sigma_\star))$; (ii) $\gamma_\uparrow$ is the pointwise-greatest element of that solution set: for every solution $\gamma$ and every variable $x$, $\gamma(x) \sqsubseteq \gamma_\uparrow(x)$.

*(Scope notes. First: solved form here is taken with the RE-same-rot deferrals discharged — an in-flight deferred equation $x \cdot t = s \cdot x$ is satisfied by $\gamma_\uparrow$ iff $s = t$, so a surviving $s \ne t$ deferral postpones (i) to the closing step that resolves or fails it; the success direction of Cor. 5.5 is unaffected, since an end-to-end successful run has discharged its deferrals. Second: (ii)'s greatestness is over the solution set of the* final configuration, *which encodes the run's placement commitments; it does not by itself give greatestness over $\mathrm{Sol}(\Phi_0)$ — that transfer is the content, and the caveat, of Prop. 6.4.)*

**Corollary 5.5 (Decision).** A fair run fails iff the *policy-strengthened* system is unsatisfiable — $\Phi_0$ plus Lemma 5.1's policy commitments (placements from RE-open-closed and RE-nested, and deferred-equation resolutions from RE-cross and RE-same-rot). Success implies $\mathrm{Sol}(\Phi_0) \ne \emptyset$ outright (Prop. 5.4 composed back through the preservation chain — the success direction is unconditional). Failure implies $\mathrm{Sol}(\Phi_0) = \emptyset$ *when the failing step is semantic*; a policy rejection can fail an $\approx$-satisfiable input (Remark 2.11's witness). On inputs satisfiable *and* placement-undiscriminated, runs terminate by Thm. 5.2(a) and cannot fail, so they decide positively. On unsatisfiable inputs, runs cannot succeed; that they *terminate* — making the solver a decision procedure rather than a semi-decision procedure — is exactly the Detection Lemma of Thm. 5.2(b).

**Theorem 5.6 (Soundness, representation, most generality).** Let a fair run on finite $\Phi_0$ terminate in $\langle\emptyset; \sigma_\star; B_\star\rangle$. Then, with $V = \mathrm{vars}(\Phi_0)$:

1. *(Representation)* $\gamma \in \mathrm{Sol}(\Phi_0)$ iff $\gamma$ extends to a ground $\hat\gamma$ (on the fresh variables) with $\hat\gamma = \hat\gamma \circ \sigma_\star$ and $\hat\gamma \in \mathrm{Sol}(\mathrm{constr}(B_\star))$.
2. *(No guessing / entailment)* every binding $x \mapsto T$ in $\sigma_\star$ is conservative: every solution of $\Phi_0$ extends to one satisfying $x = T$.
3. *(Most generality among models)* for every substitution $\sigma \models \Phi_0$ there is $u$ with $u \circ \sigma_\star = \sigma$ on $V$ — indeed $u = \sigma$ works: $\sigma = \sigma \circ \sigma_\star$ on $V$.

**Caveat to the whole theorem (the $\approx$/policy weakening, with Def. 3.4's adoption):** (1)'s "iff" and (2)'s "every solution extends" hold exactly for runs of semantic steps; across Lemma 5.1's *policy steps* the forward direction holds after re-placing the policy-committed markers, hence unconditionally only on placement-undiscriminated inputs (the conjectured surface fragment — Cor. 5.5). (3) is up to $\approx$ as the proof now states: $\sigma_\star$ is most general in content; its marker placements are the policy's, not canonical. This is the theorem-level form of the bullet bitten in Remark 2.15(v).

**Caveat to (1)/(2):** the extension over fresh variables is existential, exactly as in standard unification with fresh-variable introduction; the answer is unique up to renaming of fresh variables.

**Corollary 5.7 (Forced variables).** $\sigma_\star$ binds only variables whose values (relative to the remaining free ones) are entailed by $\Phi_0$ — by 5.6(2). Conversely every variable left unsolved genuinely admits at least two solutions or carries only its top default ($\gamma_\uparrow$ vs. e.g. committing a cap), by Prop. 5.4 and inspection of solved form.

**Corollary 5.8 (Principal model, recovered).** If $B_\star$ stores no atom caps and no row caps (adjacencies allowed), then $\sigma_\star$ (read as a model, free variables universal) satisfies $\sigma_\star \models \Phi_0$, and by 5.6(3) it is the $\le$-least model: the blog's "principal model" claim holds exactly in this case. Example 3.5 shows the restriction is necessary.

### 6. Closing

**Definition 6.1 (Closing policy).** Variables carry a provenance tag: *leaf* (belonging to a terminal tensor's shape; a subset are *parameters*) or *interior*. Closing runs after solving, so dimension leaves close to the $\mathrm{glb}$ defined by the saturated store of Def. 4.1. Closing then proceeds interleaved:
1. *(Close leaves, downward.)* For each unsolved leaf variable: bind a dimension variable to its stored atom cap if any, else to $1_\emptyset$ — except a **parameter** with no cap at some position: $\mathsf{error}$ ("unspecified hidden dimension"). Bind a row variable to the join (Prop. 2.7) of the *ground parts* of its stored caps, with *holes* (positions where no cap is ground) filled by $1_\emptyset$, and no-further-axes beyond the caps' extent; parameters error at holes.
2. *(Re-solve.)* The bindings re-emit the variables' stored constraints (Def. 4.1); run the solver to a new solved form.
3. *(Close interiors, upward.)* Bind every remaining variable to its top ($1_\emptyset$ / the empty row); re-solve once more (re-emissions against tops discharge by DI-top and Prop. 2.6, so this cannot fail).

The downward leaf choice is local: with fully ground caps it is the join of the stored lower bounds, i.e. the $\sqsubseteq$-least value satisfying those caps alone. A row cap with holes is a policy guess, since filling a hole with $1_\emptyset$ is generally incomparable with filling it by a concrete atom; parameter errors are the guard against making that guess silently.

**Proposition 6.2 (Closing terminates and is sound).** The interleave terminates, and if no $\mathsf{error}$/$\mathsf{fail}$ occurs, the resulting total ground substitution $\gamma_{\mathrm{close}} \in \mathrm{Sol}(\Phi_0)$. Ground commitments add no variable-to-variable rank facts, so the variable–variable fragment keeps its model and bounds all growth (the deficit potential consults only open-result lineages); a rank-breaking commitment therefore fails *finitely*, through the absolute fragment. The Detection Lemma is not needed here.

**Remark 6.3 (Transitive $\mathrm{glb}$ and incomplete leaf closing).** Let
$$
\Phi_1 = \{3_b \sqsubseteq \alpha,\ \alpha \sqsubseteq \beta,\ 5_b \sqsubseteq \beta\}
$$
with $\alpha,\beta$ leaves. By Def. 4.1, $3_b \sqsubseteq \alpha \sqsubseteq \beta$ contributes the lower bound $3_b \sqsubseteq \beta$, so $\mathrm{glb}(\beta)=3_b \vee 5_b = 1_\emptyset$ by DI-cap. Thus solving collapses $\beta$ to $1_\emptyset$ before leaf closing commits anything. The solutions of $\Phi_1$ are exactly $\beta = 1_\emptyset$, $\alpha \in \{3_b, 1_\emptyset\}$, and deterministic closing selects $\alpha=3_b,\beta=1_\emptyset$ in every tested emission order; the branch $\beta \mapsto 5_b$ is not a legal close-to-$\mathrm{glb}$ run.

This order-independence is not completeness. Let
$$
\Phi_2 = \{3_b \sqsubseteq \alpha,\ 5_b \sqsubseteq \beta,\ \gamma \sqsubseteq \alpha,\ \gamma \sqsubseteq \beta\}
$$
with $\alpha,\beta$ leaves and $\gamma$ interior. The store is satisfiable, e.g. $\alpha=\beta=1_\emptyset,\gamma=3_b$. Leaf-downward closing commits the capped leaves to their saturated GLBs, $\alpha \mapsto 3_b$ and $\beta \mapsto 5_b$; re-solving then pins $\gamma$ below two distinct atoms and fails. This is a policy rejection: satisfying the store requires relaxing at least one capped leaf upward to $1_\emptyset$, and the deterministic closing policy deliberately does not backtrack to larger leaf shapes.

**Proposition 6.4 (Uniform-upward closing yields the greatest solution).** $\gamma_\uparrow$ of Prop. 5.4 is a solution of $\Phi_0$ — unconditionally: the soundness direction of Lemma 5.1 composes through policy steps. Its *greatestness* needs more care than the original one-line proof ("Prop. 5.4 plus Theorem 5.6(1)") admitted: that route uses 5.6(1)'s only-if direction, which policy steps break — the final configuration encodes the run's placement commitments, so Prop. 5.4(ii) quantifies over a possibly *proper* subset of $\mathrm{Sol}(\Phi_0)$. Three correct forms:

(i) *Policy-relative, marked order.* $\gamma_\uparrow$ is pointwise-greatest among the solutions of the *policy-strengthened* system — $\Phi_0$ plus the run's placement commitments; equivalently, the solutions that extend into the final configuration (Cor. 5.5's phrase). On placement-undiscriminated inputs (conjecturally the whole surface fragment) this set is all of $\mathrm{Sol}(\Phi_0)$ and the unqualified claim holds.

(ii) *The unqualified marked claim is false.* $\Phi_0 = \{\langle\rho\rangle \approx [\,]\cdot\diamond\cdot[3,5]\}$: the run commits $\rho \mapsto [\,]\cdot\diamond\cdot[3,5]$ (inherit-the-split), so $\gamma_\uparrow(\rho) = [\,]\cdot\diamond\cdot[3,5]$, while $\gamma(\rho) = [3]\cdot\diamond\cdot[5] \in \mathrm{Sol}(\Phi_0)$ under Def. 3.4's $\approx$-semantics is $\sqsubseteq$-*incomparable* to it (Prop. 2.10(ii)).

(iii) *Absolute up to $\approx$, relative to the rank policy.* For every $\gamma \in \mathrm{Sol}(\Phi_0^{\mathrm{rk}})$ — $\Phi_0$ plus the run's deferred-equation resolutions, the rank-level policy commitments of (c-split)/(d) — and every $x \in V$: $\gamma(x) \sqsubseteq^{1} \gamma_\uparrow(x)$ at row sort, plain $\sqsubseteq$ at dimension sort. **[proved — A.6]** Placement commitments need no exclusion: $\sqsubseteq^1$ is marker-blind on its lower side, which is exactly why the one-shot order is the right comparison. Rank commitments do: a solution realizing a foreclosed sporadic/conjugate branch can have smaller rank at a variable than $\gamma_\uparrow$'s committed value, and no marker-blind order repairs a rank gap. On deferral-free runs $\Phi_0^{\mathrm{rk}} = \Phi_0$ and the statement is unconditional.

**Remark 6.5 (Non-principality witnessed).** $\Phi = \{3_b \sqsubseteq \alpha\}$, $\alpha$ a leaf: $\gamma_{\mathrm{close}}(\alpha) = 3_b$; $\gamma_\uparrow(\alpha) = 1_\emptyset$; both are solutions, neither factors through the other by substitution (Example 3.5). Closing selects by policy, justified by allocation (leaves) and parsimony (interiors), not by entailment.

### 7. Projection inference

Throughout, fix one operation; its shapes are solved and **closed** (ground). Its constraint set $\Phi_{\mathrm{op}}$ is re-derived with **freshened projection identifiers**: each axis of each participating tensor occurrence carries an id $p \in P$, injectively, used by this operation only; $\mathrm{sz}(p) \in \mathbb{N}_{\ge 1}$ is the axis's solved size. (The spec's fresh row variables are eliminated by the local re-solve — Lemma 7.7 — and its label variables resolve to operand axes, inheriting ids.)

**Definition 7.1 (Projection language).** Atoms $q ::= \mathsf{Proj}(p) \mid \mathsf{Sol}(\mathit{idx})$, where $\mathit{idx}$ is an externally supplied index: $\mathsf{Fix}(c)$ or an external symbol. Equations $e ::= \mathsf{Eq}(q_1, q_2) \mid \mathsf{Iter}(q)$, with $\mathsf{Eq}$ symmetric.

**Definition 7.2 (Elaboration $\llbracket\cdot\rrbracket$).** From the ground $\Phi_{\mathrm{op}}$:
P1. $d_1 = d_2 \rightsquigarrow \mathsf{Eq}(\llbracket d_1\rrbracket, \llbracket d_2 \rrbracket)$.
P2. $d_{\mathrm{res}} \sqsubseteq d_{\mathrm{op}}$ with $\mathrm{sz}(d_{\mathrm{op}}) = 1 \rightsquigarrow \mathsf{Eq}(\llbracket d_{\mathrm{op}}\rrbracket, \mathsf{Sol}(\mathsf{Fix}\,0))$ (sever and pin).
P3. $d_{\mathrm{res}} \sqsubseteq d_{\mathrm{op}}$ otherwise $\rightsquigarrow \mathsf{Eq}(\llbracket d_{\mathrm{res}}\rrbracket, \llbracket d_{\mathrm{op}}\rrbracket)$.
P4. $R_1 \approx R_2$: outer-edge alignment; P1 per aligned pair (ranks match — shapes are closed and the equality held).
P5. $R_{\mathrm{res}} \sqsubseteq R_{\mathrm{op}}$: outer-edge alignment; P2/P3 per aligned pair, with P2's severance applied row-wise (result side of the pair $\rightsquigarrow \mathsf{Iter}$, operand pinned); each surplus interior result axis $\rightsquigarrow \mathsf{Iter}(\llbracket\cdot\rrbracket)$.
P6. each terminal axis $\rightsquigarrow \mathsf{Iter}(\llbracket\cdot\rrbracket)$.
Side condition: $\mathsf{Sol}(\mathsf{Fix}\,c)$ is emitted only with $0 \le c < $ the axis's size (slices are validated against solved sizes).

**Definition 7.3 (Solver).** A single pass over the finite equation set $E$ maintaining: a union–find partition of $P$; a partial pin map on classes; an iterate set $I$ of classes. $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Proj}\,q)$: union, $\mathsf{fail}$ if class sizes differ. $\mathsf{Eq}(\mathsf{Proj}\,p, \mathsf{Sol}\,i)$: pin $p$'s class at $i$, $\mathsf{fail}$ on a conflicting pin. $\mathsf{Eq}(\mathsf{Sol}\,i, \mathsf{Sol}\,j)$: check $i = j$. $\mathsf{Iter}(\mathsf{Proj}\,p)$: add class to $I$. $\mathsf{Iter}(\mathsf{Sol}\,\_)$: no-op. **Labeling:** pinned class $\mapsto$ its pin (pins dominate $I$); unpinned class in $I$ with size $> 1$ $\mapsto$ a fresh iterator symbol; otherwise $\mapsto \mathsf{Fix}(0)$. The **product space** $P^{\times} = \prod_j R_j$, one factor $R_j = \{0..n_j{-}1\}$ per fresh iterator with $n_j$ the class size; the **index map** $\pi_T$ of tensor $T$ maps each axis to its class's label.

**Theorem 7.4 (Canonicity — Theorem 2 of the template). [proved]** For a finite $E$: the partition, the pin map, the iterate set, and the labeling are independent of processing order; the solver fails iff (a) some $\approx$-class (the least equivalence containing the $\mathsf{Eq}(\mathsf{Proj},\mathsf{Proj})$ pairs) contains two ids of different sizes, or (b) some class receives two distinct pins, or (c) some $\mathsf{Eq}(\mathsf{Sol}\,i, \mathsf{Sol}\,j)$ has $i \ne j$. On success the output is unique up to a bijective renaming of the fresh iterator symbols. The solver resolves no ambiguity: it has no default, no direction, and no policy.

*Proof.* $\approx$ is a closure operator's value — the least equivalence containing the given pairs — hence order-independent; union–find computes exactly it. *Size failure:* if a class has two sizes, then along any processing order some union merges sub-classes of different recorded sizes (sizes are constant per id; a class's recorded size is any member's, well-defined per sub-class by induction), so failure occurs; conversely if all classes are size-uniform no union can fail. *Pins:* the pin map on classes is the lift of the partial function induced by the $\mathsf{Eq}(\mathsf{Proj}, \mathsf{Sol})$ equations along $\approx$; failure iff the lift is inconsistent — a property of $(E, \approx)$, not of order. (c) is a per-equation check. The iterate set is the $\approx$-saturation of the marked ids — again a closure. The labeling is a function of (partition, pins, $I$, $\mathrm{sz}$); the only free choice anywhere is the identity of the fresh symbols, one per labeled class, whence uniqueness up to bijection. $\square$

**Definition 7.5 (Reduction; coverage).** A factor $j$ of $P^\times$ is a *reduction axis* iff no component of $\pi_{\mathrm{LHS}}$ is $\mathsf{Iter}(s_j)$.

**Proposition 7.6 (Injectivity, surjectivity, and the reduction characterization). [proved]** With core labels only ($\mathsf{Iter}/\mathsf{Fix}$):
(i) $\pi_{\mathrm{LHS}} : P^\times \to \mathrm{Addr}$ is injective iff every product variable occurs in some component of $\pi_{\mathrm{LHS}}$ — i.e. iff there is no reduction axis. *Proof:* ($\Leftarrow$) the components mentioning each $s_j$ reconstruct the point. ($\Rightarrow$) a missing $s_j$ has $n_j \ge 2$ (factors are minted only at size $> 1$), so two points differing only in $s_j$ collide.
(ii) $\pi_{\mathrm{LHS}}$ is surjective onto the LHS's index domain iff every LHS axis of size $> 1$ is labeled $\mathsf{Iter}$ and the map (axis $\mapsto$ its variable) is injective on those axes. *Proof:* ($\Leftarrow$) given a target multi-index, set each named variable by its (unique) axis, others arbitrarily; $\mathsf{Fix}(0)$ axes have size $1$ by the labeling rule (an unpinned size-$1$ class) or are pinned slices, excluded on the LHS in the core. ($\Rightarrow$) a size-$>1$ axis labeled $\mathsf{Fix}$ misses indices; two axes sharing $s$ hit only the diagonal.
(iii) Hence: the accumulating read-modify-write is required iff a reduction axis exists (non-injectivity), and pre-initialization is required iff $\pi_{\mathrm{LHS}}$ is non-surjective or (under an erasing accumulator) non-injective — the `=:+` analysis, now a corollary.

**Lemma 7.7 (Local re-solve succeeds; locality). [sketch]** If the global solve and closing succeeded, then for each operation: (a) the per-operation re-derivation (spec template with fresh row/dimension variables against the ground operand shapes) has a solution, and the core solver finds it, eliminating every spec variable; (b) the resulting $\approx$ depends only on $\Phi_{\mathrm{op}}$ — by construction of the freshened ids, no constraint of any other operation mentions any id of $P$, so no external fact can enter the closure. *Proof sketch for (a):* the operation's constraints are a subset of $\Phi_0$ up to the freshening bijection on ids (which does not affect sizes) and renaming of spec variables; $\gamma_{\mathrm{close}}$ restricted along that bijection is a solution; by Corollary 5.5 the solver cannot fail on a satisfiable set, and at ground shapes the row alignments pin every spec row variable, after which every label variable is unified with a ground operand axis. **[The bijection/subset bookkeeping should be spelled out against the actual elaboration in `shape.ml`; flagged for the full paper.]**

**Theorem 7.8 (Soundness of projections w.r.t. shapes — Theorem 3 of the template). [proved given 7.7]** Let shapes be solved and closed, the per-operation elaboration as in Def. 7.2 with its side condition, and the local solve succeed. Then:
1. *(Bounded addressing)* for every participating tensor $T$ and every $p \in P^\times$, $\pi_T(p)$ is a valid multi-index: componentwise, an $\mathsf{Iter}(s_j)$ component addresses an axis whose size equals $n_j$, and $s_j < n_j$; a $\mathsf{Fix}(c)$ component has $c <$ the axis size.
2. *(Co-iteration is finer than size-equality)* $p \approx q \Rightarrow \mathrm{sz}(p) = \mathrm{sz}(q)$, and $\approx$ is the **least** equivalence justified by this operation's own constraints (Thm. 7.4 + Lemma 7.7(b)) — in particular, axes identified in size by other operations' constraints are not co-iterated here.

*Proof.* (2) Size-uniformity of classes is the solver's invariant (Thm. 7.4(a)); leastness is closure; locality is Lemma 7.7(b). (1) $\mathsf{Iter}$: the factor's range was minted with the class size, the class size equals each member axis's solved size by (2), and $s_j$ ranges over $\{0..n_j{-}1\}$. $\mathsf{Fix}(0)$ from the labeling: the axis's class has size $1$... — careful: an *unpinned, uniterated* class of size $>1$ also labels $\mathsf{Fix}(0)$ by the rule; $0 <$ size always, so addressing is valid (such an axis is read at $0$ — this is the broadcast severance and the no-partner case, both intended). Pinned $\mathsf{Fix}(c)$: the side condition of Def. 7.2. External symbols: validity is the caller's contract (a kernel argument), recorded as a premise. $\square$

**Remark 7.9 (Asymmetry of the two passes, now formal).** Shape solving needs a closing *policy* (Def. 6.1; non-principal by Remark 6.5). Projection solving needs only a *naming* (Thm. 7.4: forced, unique up to renaming, no direction). The slogan "more principal than shape inference" is these two statements side by side.
