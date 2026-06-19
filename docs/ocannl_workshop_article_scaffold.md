# Neural Networks Without Shape Boilerplate: An OCaml DSL Case Study

> Scaffold / template for rewriting by hand.  
> Working subtitle option: **Shape and projection inference for staged tensor programs**.

## Metadata / positioning notes

**Primary venue:** OCaml Workshop.  
**Possible handoff fit:** ML Family Workshop.  
**Likely category:** experience report / demo / implementation talk, with a theoretical core.  
**Primary claim:** OCANNL lets realistic neural-network code be written in OCaml without explicit shape plumbing, while preserving enough inferred structure to lower to explicit loop nests.  
**Non-claim:** this submission does not claim hardware-saturating performance or parity with PyTorch/JAX/vendor kernels.

Possible note to chairs:

> This proposal is intended primarily as an OCaml DSL/compiler implementation report. It may also fit the ML-family workshop because the central technical contribution is a shape/projection inference architecture for staged ML-embedded tensor programs.

---

# 0. Abstract draft to rewrite

Neural-network code often contains a large amount of incidental shape manipulation: reshapes, singleton axes, transposes, explicit reductions, and comments explaining which dimension is which. OCANNL is an OCaml-embedded DSL that tries to remove that boilerplate. Users write tensor algebra with ordinary OCaml functions and small syntax extensions; the system infers both tensor shapes and the loop-index maps used for code generation.

The central idea is that one elaborated constraint set has two readings. Shape inference reads the constraints as a broadcast-aware order over rows of axes. Projection inference rereads them locally as an equivalence problem over axis identities, producing loop variables, fixed broadcast indices, and affine index maps. Reductions then emerge from output independence, and convolution is represented as contraction with affine operand addressing.

The paper presents this design through two examples: core multi-head attention and rank-polymorphic `conv2d`. It then states the formal core results that anchor the implementation: the non-distributive broadcast lattice, the corrected substitution-plus-residual-store shape-solving theorem, the rank-cycle termination check, projection canonicity, and projection soundness. The implementation status section shows that, before hardware scheduling, the compiler already lowers the high-level notation to clean scalar loop code after virtualization, inlining, simplification, and common-subexpression elimination.

---

# 1. Introduction: the shape-boilerplate problem

## 1.1 Opening problem

Start with the practical problem, not the formalism.

Possible opening shape:

> Modern neural-network code often alternates between arithmetic and shape bookkeeping. Attention code in mainstream tensor libraries is full of `reshape`, `view`, `transpose`, `unsqueeze`, `expand`, and integer-axis arguments. These operations are rarely the mathematics of the layer; they are how the programmer convinces the library that the intended loops line up.

Then state OCANNL’s goal:

> OCANNL asks whether an OCaml DSL can let the user state the algebra and let the compiler infer the missing shapes, broadcasts, contractions, affine addresses, and loop maps.

Explicitly avoid overclaiming:

> The aim in this paper is not to claim competitive kernel performance. It is to show that the frontend and middle-end recover enough structure to compile realistic neural-network fragments without shape boilerplate.

## 1.2 Contribution paragraph

Write as one paragraph or a short list.

Suggested content:

1. A surface OCaml DSL for neural-network expressions using `%op`, ordinary functions, and compact tensor specifications.
2. A two-reading inference architecture: shape constraints first solve sizes and row structure; the same operation constraints are then reread locally to infer loop-index maps.
3. Formal core results: non-distributive broadcast lattice; solver answer as substitution plus residual store rather than a principal ground model; rank-cycle termination check; canonical projection inference; projection soundness.
4. Implementation evidence from multi-head attention, rank-polymorphic convolution, and generated single-threaded scalar code before scheduling.

---

# 2. OCANNL by example

This section should be code-first. The reader should see why the machinery exists before seeing the machinery.

## 2.1 Figure 1: core multi-head attention

Use a simplified but compiling variant of the full implementation.

```ocaml
let%op multi_head_att ~num_heads ~d_k ~d_v () x =
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

  let attn_weights =
    softmax ~spec:" ... | t -> ..." () scores
  in

  { w_o } *
    (attn_weights +* v
       " ... s | t -> h; ... t | h e => ... s | h e"
       [ "e" ])
```

Possible caption:

> **Figure 1. Core multi-head attention in OCANNL.** The code names only semantic axes: query position `s`, key/value position `t`, head `h`, key width `d`, and value width `e`. Batch rank, parameter shapes, score shape, contractions, and loop-index maps are inferred.

Explain only the essential pieces:

- `q`, `k`, `v` are ordinary OCaml bindings of tensor expressions.
- The first `+*` compares every query position `s` with every key position `t`, per head `h`, contracting over `d`.
- The second `+*` uses attention weights over `t` to combine values, preserving `s`, `h`, and `e`.
- `Shape.set_dim h/d/e` pins semantic dimensions; these are not full tensor-shape annotations.
- The output width of `w_o` is intentionally left for the surrounding context, typically a residual connection, to determine.

Suggested prose:

> The important part of the example is what is absent. The code does not explicitly reshape a model-width axis into `(heads, per_head_width)`, transpose the key tensor, create singleton axes for broadcasting, or list an integer softmax axis. The user names semantic axes and the compiler infers how the operation iterates.

## 2.2 Figure 2: rank-polymorphic `conv2d`

Use this as the second code figure. It supports the technical claim that OCANNL infers index maps, not merely sizes.

```ocaml
let%op conv2d ~label ?(kernel_size = 3) ?(stride = 1)
    ?(use_padding = true) ?out_channels () x =
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  Option.iter out_channels ~f:(Shape.set_dim oc);
  x
  +* { kernel }
       "... | stride*oh+kh, stride*ow+kw, ..ic..;
             kh, kw, ..ic.. -> ..oc..
        => ... | oh, ow, ..oc.."
       [ "kh"; "kw"; "oc" ]
  + { bias = 0. }
```

Possible caption:

> **Figure 2. Rank-polymorphic 2D convolution.** The input is addressed at affine spatial positions `stride*oh+kh` and `stride*ow+kw`. The kernel axes `kh` and `kw`, together with the input-channel row `..ic..`, are contracted. The context row `...` and output-channel row `..oc..` are inferred and may have arbitrary rank.

Suggested explanation:

> In the usual CNN case, `..ic..` and `..oc..` each instantiate to one channel axis. The operator does not require that. It treats channels as rows of axes: a single channel axis is one instantiation, but a structured channel block with several axes is also admissible. This is rank polymorphism at the channel block, not just at the batch/context prefix.

Key point to stress:

> Convolution is the same map-reduce structure as a contraction. The new ingredient is affine operand addressing: an input cell is read at a position computed from output and kernel loops.

---

# 3. One elaboration, two readings

This is the conceptual bridge from examples to theory.

Include a figure like this:

```text
OCaml tensor expression
        ↓ elaboration
shape constraints
        ↙                         ↘
shape reading                      projection reading
broadcast order                     local equivalence
sizes + residual store              loop classes + index maps
        ↓                         ↓
closed tensor shapes                product space
                 ↘               ↙
                    loop IR
```

Possible paragraph:

> OCANNL deliberately separates two questions that many tensor APIs conflate. Shape inference asks how large each axis is and where rank-polymorphic rows close. Projection inference asks which loops index which axes in this particular operation. Equal size is global; co-iteration is local. Two axes may have the same size because of another operation elsewhere in the program, but that does not mean they should share a loop here.

Mini table:

| User-level phenomenon | Shape reading | Projection reading |
|---|---|---|
| pointwise broadcast | result refines operand | operand axis pinned to `0` |
| matrix multiply | same labels imply equal dimensions | output omits one loop variable, hence reduction |
| convolution | affine size relation | operand addressed by affine expression |
| concatenation / split | summed extent, with zero as coproduct unit | coupled factor, traversed sequentially |

Probably keep concatenation in the table, but not as a main proof obligation in this paper.

---

# 4. Shape inference, briefly but formally

Do not reproduce the whole formal core. Include the theory that explains the design.

## 4.1 Dimensions and the broadcast order

Possible wording:

> A dimension is either the claim-free unit `1_emptyset` or a concrete positive size carrying a basis tag. The order is `d₁ ⊑ d₂` iff `d₂` is `1_emptyset` or `d₁ = d₂`. Thus `1_emptyset` is the top: it makes no claim and may broadcast to any concrete axis. Distinct concrete sizes are incomparable atoms; their meet is broadcast error and their join is the claim-free top.

Theorem box:

> **Result 1 — Non-distributive broadcast lattice.** The dimension order, completed with an error bottom, is a bounded lattice. With at least three concrete atoms, it contains the diamond `M₃`, hence is non-distributive.

Consequence:

> This is why OCANNL does not simply reuse distributive-lattice algebraic-subtyping machinery. Tracking concrete sizes in the solver creates the antichain that destroys distributivity.

## 4.2 Rows and rank polymorphism

Possible wording:

> A row is a sequence of dimensions with a marker. The marker is where rank broadening inserts claim-free units and where row variables are spliced. This makes rank polymorphism part of the row order rather than a separate type-scheme mechanism.

Mention only what is needed:

- rows have leading and trailing flanks;
- rank growth happens at the marker;
- `...` and `..ic..` are row variables in this structure;
- ordinary OCaml function application re-emits constraints at call sites, giving call-site polymorphism without a full scheme language.

## 4.3 Corrected shape-solving result

This should be included because it demonstrates that the formalization changed the claims.

Theorem box:

> **Result 2 — Shape solving returns a symbolic answer, not always a principal ground model.** In the core fragment, solving returns a substitution plus a residual bound store. This pair represents the solution set without guessing and is most general up to the row-equivalence used for closed-row equality. A principal ground model is recovered exactly when no residual caps remain.

Illustrative example, one sentence:

> A residual cap such as `3 ⊑ α` has incomparable minimal ground choices, `α = 3` and `α = 1_emptyset`, so a single least committed ground model does not exist.

## 4.4 Termination and the rank-cycle check

Include this as an implementation lesson.

Theorem/lesson box:

> **Result 3 — Termination requires a rank-cycle check.** Mutually-growing row variables can make the naive deficit rule diverge. The implementation now maintains a persistent graph of entailed row-rank facts and rejects positive-weight cycles.

Suggested prose:

> This is a useful example of theory changing the implementation rather than merely explaining it after the fact: the formal core identified a real divergence case in row-rank cycles, and the implementation was updated with a persistent `global_rank_edges` check.

---

# 5. Projection inference

## 5.1 Why shape equality is not loop identity

Possible wording:

> Shape inference is global: it solves sizes for the whole expression graph. Projection inference is local: it recomputes loop identities per operation with fresh projection identifiers. This enforces the discipline “global sizes, local identities.”

Explain the algorithm compactly:

1. Freshen each participating axis with an operation-local projection identifier.
2. Re-derive this operation’s constraints from the spec and solved shapes.
3. Read equalities as union operations.
4. Read genuine broadcasts as pins to `Fix(0)`.
5. Add iterator markers for axes that must be traversed.
6. Label each union-find class as pinned, iterated, or fixed.

## 5.2 Theorem boxes

> **Result 4 — Projection canonicity.** Given solved and closed shapes for one operation, core projection inference is forced and unique up to renaming of fresh iterator symbols. It has no closing direction and no defaulting policy.

> **Result 5 — Projection soundness.** If shape solving and closing succeed for an operation, the inferred index maps address within the solved tensor shapes. Co-iteration is finer than size equality: axes co-iterate only when this operation’s own constraints force them to.

## 5.3 Attention revisited

Return to Figure 1 and explain:

- In the score computation, `d` occurs in `q` and `k` but not in the output map; therefore `d` is a reduction axis.
- `s` and `t` are independent product axes, producing all query/key pairs.
- `h` co-iterates because it is present on both sides and in the result.
- Batch/context axes hidden behind `...` are transported polymorphically.

Possible phrasing:

> The attention spec never says “reduce over `d`.” It says how axes coincide. The reduction follows because the output map is independent of the loop variable for `d`.

---

# 6. Convolution as affine-indexed contraction

Use this as the section that justifies `conv2d` as more than another example.

Possible wording:

> The loop nest for convolution and the loop nest for a matrix-vector contraction have the same structure. Both iterate an output position and a contracted axis; both accumulate into an output cell. The difference is that convolution reads its input at an affine expression of the loops, such as `o + k`, rather than at a single loop variable.

Include pseudo-loop comparison:

```text
matrix-vector:
for i:
  out[i] = 0
  for j:
    out[i] += a[i, j] * v[j]

valid convolution:
for o:
  out[o] = 0
  for k:
    out[o] += x[o + k] * kernel[k]
```

Then tie to Figure 2:

> In `conv2d`, `oh` and `ow` are output loops, `kh` and `kw` are kernel loops, and the input spatial axes are addressed by affine expressions. The channel row `..ic..` is contracted; `..oc..` is produced.

Mention theorem status carefully:

> The paper’s fully proved core excludes `Affine` and `Concat`. Affine indexing is implemented and fits the same staged pattern: first infer base co-iteration classes, then evaluate derived affine index expressions against the iterators minted by the first stratum. The expected determinacy result is that the extension still resolves no ambiguity, but conflicts between derived terms may surface in the later stratum.

---

# 7. Lowering quality before scheduling

This section answers the performance/benchmark question without making performance the central claim.

## 7.1 What to show

Show one generated-code excerpt, preferably from a small convolution or attention subexpression.

Caption idea:

> **Generated single-threaded code after virtualization, inlining, simplification, and CSE.** This is a lowering-quality sanity check, not a hardware-performance benchmark.

Suggested prose:

> OCANNL currently emits single-threaded kernels by design sequencing. The important observation at this stage is that the high-level, shape-inferred notation lowers to ordinary loop code and does not leave a pile of tensor-level abstraction debris. Virtual nodes are inlined into consumers; affine index maps compose; repeated scalar work is collapsed by common-subexpression elimination; small transients can drain into local-scope scalar temporaries.

## 7.2 Performance boundary paragraph

Use something close to this:

> This submission does not claim competitive runtime performance. The current implementation focuses on the frontend and middle-end: shape inference, projection inference, lowering to loop nests, and abstraction removal before scheduling. The performance-critical layer — tiling, vectorization, thread/block mapping, shared memory, tensor cores, and search over schedules — is the next stage. Early runtime measurements without that layer would mostly measure the absence of scheduling. The talk will report preliminary numbers where available, especially inference overhead and generated-loop structure, and will label runtime numbers as pre-schedule baselines.

## 7.3 Q&A-ready answer

> Tensor cores are downstream of this talk. This talk shows that OCANNL can infer the loop structure and lower it to clean scalar code. Mapping those loops to tensor cores requires schedule values, axis annotations, tiling, local/shared memory, and search. That is the next layer, not the claim of this submission.

---

# 8. Status, limits, and extensions

Keep this short. It prevents the paper from sounding stronger than it is.

## 8.1 What is proved

- Core dimension and row order facts.
- Corrected shape-solving theorem: substitution plus residual store.
- Rank-cycle check for termination, with one detection lemma still a formal gap if presenting full decidability.
- Projection canonicity.
- Projection soundness.

Use careful status labels:

> The core fragment is formalized; several proof obligations are proved, while the full termination detection lemma and surface-language marker-provenance invariant remain open. The paper reports the established core results and states the remaining obligations explicitly.

## 8.2 What is implemented beyond the core

- Affine indexing, including convolution and striding.
- Padding margins as an order-independent max-fold.
- Concatenation/split as coupled iteration over coproduct factors.
- Neural-network building blocks: MLP, attention, RoPE, transformer blocks, convolution, pooling, normalization, residual blocks, LeNet-style and mobile-CNN-style fragments.

## 8.3 What is not centered in this paper

- Full concatenation proof.
- Tensor-core scheduling.
- Autotuning/search.
- Distributed contexts / storage identity, except as related work or future direction.

---

# 9. Related work and design-space placement

This should be compact and targeted.

## 9.1 Star / algebraic shapes

Use this comparison if space permits:

> Star and OCANNL agree on the deep separation: a shape is not its index. Star types the correspondence using algebraic shapes and an index-type metafunction. OCANNL computes along the correspondence: shape inference solves sizes and projection inference constructs the loop-index maps. Star keeps a distributive lattice by not typing concrete sizes; OCANNL tracks sizes in a non-distributive solver lattice.

Possible table:

| System | What is inferred/typed | Size arithmetic | Shape/index relation |
|---|---|---|---|
| Star | algebraic shape types | runtime / not in type lattice | typed isomorphism |
| OCANNL | solver-level shapes + projections | bespoke constraint solver | computed bridge to loop maps |

## 9.2 tinygrad / rangeify

Use tinygrad mainly to position scheduling.

> tinygrad’s rangeify-era design also represents movement as loop/range structure. The remaining gap is scheduling: tinygrad has a minimal searchable schedule language over loop axes, while OCANNL currently has the inferred loop IR and clean scalar lowering but not yet the schedule-as-value layer.

## 9.3 PyTorch/JAX/etc.

Keep this short and non-adversarial.

Possible wording:

> Mainstream tensor frameworks provide mature kernels and flexible APIs, but the user-visible tensor code often contains explicit shape manipulation. OCANNL explores a different point in the design space: a staged DSL where the host language supplies ordinary abstraction and the compiler solves shape/projection structure before lowering.

---

# 10. Conclusion

Possible structure:

1. Restate the user-visible claim.
2. Restate the theoretical spine.
3. Restate the implementation status and next step.

Draft to rewrite:

> OCANNL’s surface examples are intentionally ordinary neural-network code: attention, convolution, pooling, and transformer blocks. The unusual part is what the code does not say. It does not spell out the batch rank, reshape into heads, list reduction axes, or hand-code convolution output addresses. Those facts are recovered by a two-reading constraint architecture: shapes are solved globally as sizes and rows; projections are solved locally as loop identities and index maps. The formal core clarifies which parts are canonical, which parts are policy, and where the current implementation goes beyond the proved fragment. The next stage is not more shape syntax but scheduling: mapping the inferred loop nests to vector lanes, threads, shared memory, and tensor cores.

---

# Appendix A. Suggested theorem box wording

Use at most three boxes in the main paper. Keep the rest as inline claims.

## Box 1: Shape solving

> For the core constraint language, solving either fails or returns a substitution plus residual bound store. This answer represents the solution set without guessing and is most general up to the flat row equivalence used for closed-row equality. A principal ground model exists exactly when the residual store is empty.

## Box 2: Projection inference

> For a fixed operation with solved and closed shapes, core projection inference is canonical: it is unique up to renaming of fresh iterators and resolves no ambiguity. Broadcast axes are pinned, equal axes are unioned, and iterated classes become loop variables.

## Box 3: Projection soundness

> The inferred index maps are bounded by the solved tensor shapes. Moreover, co-iteration is local: two axes co-iterate only when this operation’s constraints force them to, not merely because global shape solving gave them equal sizes.

---

# Appendix B. Figure list

1. **Core multi-head attention code** — main user-facing example.
2. **Rank-polymorphic `conv2d` code** — affine indexing and channel-row polymorphism.
3. **One elaboration, two readings** diagram.
4. Optional: **Matrix-vector vs convolution loop nests**.
5. Optional: **Generated scalar code before scheduling**.
6. Optional: **Theorem/status table**.

---

# Appendix C. Submission-risk checklist

Before submission, check:

- [ ] Does the title foreground the user-visible payoff rather than the cryptic theory slogan?
- [ ] Does the abstract avoid claiming performance parity?
- [ ] Is the core theorem status honest: proved core vs implemented/staged extensions vs open obligations?
- [ ] Is `conv2d` presented as evidence of affine projection inference, not as a performance benchmark?
- [ ] Does the generated-code section say “lowering quality before scheduling,” not “performance”?
- [ ] Is concatenation mentioned only as an extension unless there is space to explain coproduct factors cleanly?
- [ ] Is Star positioned respectfully as a neighboring design, not as a foil?
- [ ] Is tinygrad used to motivate schedule-as-value future work, not as the main comparison?
- [ ] Is the prose rewritten in the author’s own voice before submission?

---

# Appendix D. Questions for feedback after first rewrite

When you have a draft, useful feedback questions are:

1. Does the paper still read as an OCaml Workshop paper after adding the theoretical core?
2. Are the theorem statements understandable without the full formal appendix?
3. Does the attention example carry the “no shape boilerplate” claim by itself?
4. Does the `conv2d` example feel like a natural second example or a detour?
5. Is the performance paragraph defensive, or does it cleanly delimit the contribution?
6. Does the related-work section help reviewers locate the work, or does it distract from OCANNL?
