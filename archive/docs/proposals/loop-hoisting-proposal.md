# Loop Hoisting (Loop-Invariant Code Motion) for the Low-Level IR

GitHub issue: https://github.com/ahrefs/ocannl/issues/350

## Motivation

OCANNL's compiler decides whether a tensor node can be **virtualized** (inlined into its consumers, avoiding intermediate memory allocation) based on its **visit count** — how many times it's accessed during a semi-concrete interpretation run. A node accessed exactly once per position can be virtualized.

The problem: a subexpression inside a loop that doesn't depend on the loop variable gets counted as visited N times (once per iteration), even though it computes the same value every iteration. This inflated visit count prevents virtualization.

**Loop-invariant code motion (LICM)** hoists such expressions outside the loop. After hoisting, the expression is visited once, making it eligible for virtualization. This is a standard compiler optimization with well-understood semantics, directly applicable to OCANNL's IR.

The issue author also notes this entails changing the `fix`-style loop construct to a `let rec`-style construct, since the latter is more general and naturally supports named bindings at different scope levels.

## Current State

The low-level IR (`arrayjit/lib/low_level.ml`) has two main types:

**Statements (`t`):**
- `Noop`, `Comment`, `Seq(t, t)` — control flow
- `For_loop { index; from_; to_; body; trace_it }` — loops with implicit index binding
- `Zero_out(tn)`, `Set { tn; idcs; llsc }` — memory writes
- `Set_local(scope_id, scalar_t)` — local variable assignment

**Scalar expressions (`scalar_t`):**
- `Get(tn, idcs)`, `Get_merge_buffer(tn, idcs)` — tensor reads
- `Local_scope { id; body; orig_indices }` — scoped computation (result of virtualization)
- `Get_local(scope_id)` — local variable read
- `Binop`, `Unop`, `Ternop`, `Constant`, `Embed_index` — arithmetic

The optimization pipeline in `optimize_proc` (line 1238):
1. `visit_llc` — semi-concrete interpretation, counts accesses per tensor node per index position
2. `virtual_llc` — decides virtualization based on visit counts, registers computations
3. `cleanup_virtual_llc` — removes invalid virtualizations
4. `simplify_llc` — algebraic simplification, dead code elimination

**Key observation:** There is no LICM pass. The `For_loop` construct uses a `fix`-style representation where the loop variable is implicitly bound — there is no mechanism to name and place intermediate results at different scope levels.

Key files:
- `arrayjit/lib/low_level.ml` — IR types (lines 33–65), visit counting (`visit_llc`, line 257), optimization pipeline (`optimize_proc`, line 1238)
- `arrayjit/lib/low_level.mli` — public interface
- `arrayjit/lib/c_syntax.ml` — C code generation for each IR construct
- `arrayjit/lib/indexing.ml` — loop index symbols (`Indexing.symbol`)

## Proposed Change

Add a LICM pass that runs **before** `visit_llc` in the optimization pipeline, so that hoisted expressions get visit count 1 instead of N.

This requires:

**1. Extend the IR with a `Let` binding construct** (or convert to `let rec` style as suggested). The new construct names an intermediate result and places it at a specific scope level. During code generation, this becomes a local variable declaration before the loop.

**2. Implement loop-invariance analysis.** A scalar subexpression inside a `For_loop` is loop-invariant if it does not depend on the loop's `index` symbol. This is a straightforward syntactic check over `Embed_index`, `Get` indices, and transitive dependencies. Side-effecting statements (`Set`, `Zero_out`) are never loop-invariant.

**3. Implement the hoisting transformation.** Walk the IR looking for `For_loop` nodes. For each, identify invariant subexpressions in the body, replace them with `Get_local` references, and wrap the loop with `Let` bindings for the hoisted values. Apply recursively for nested loops.

**4. Integrate into the pipeline.** Insert the LICM pass between the current start of `optimize_proc` and the `visit_llc` call:

```
LICM pass (new) → visit_llc → virtual_llc → cleanup → simplify
```

**5. Update all consumers of the IR.** The new `Let` construct must be handled in:
- `visit_llc` (visit the value once, then the body)
- `virtual_llc`, `cleanup_virtual_llc`, `simplify_llc`
- `c_syntax.ml` (generate `type var = value;` before the loop)
- Any other pattern matches on `t`

**Edge cases to consider:**
- **Write dependencies:** A subexpression that reads a tensor written inside the loop (accumulation pattern) must not be hoisted, even if its index doesn't use the loop variable
- **Nested loops:** An expression invariant to the inner loop but variant in the outer loop should be hoisted to between the two loops
- **`trace_it = false` loops:** LICM should still apply (the optimization is independent of tracing)
- **Shared subexpressions:** If the same invariant expression appears multiple times, it should be hoisted once (related to CSE, issue #351)

## Scope

**In scope:**
- Adding `Let` (or `Let_rec`) construct to the IR
- LICM analysis and transformation pass
- Pipeline integration before visit counting
- Code generation for the new construct
- Updating all pattern matches across `low_level.ml` and `c_syntax.ml`

**Out of scope:**
- Full `fix` → `let rec` refactoring of the entire IR (can be a follow-up; the initial LICM can use a simpler `Let` construct)
- Common subexpression elimination (#351 — LICM creates the infrastructure that CSE can build on)
- Local_scope initialization tracking (#340 — LICM creates more `Local_scope` nodes; the init tracking optimization from #340 applies to these but is separate work)

**Related tasks:**
- gh-ocannl-340: Track `Local_scope` initialization needs — benefits from more virtualizations enabled by LICM
- gh-ocannl-351: Common subexpression elimination — the `Let` construct from LICM is directly reusable
- gh-ocannl-311: `-march=native` — complementary; LICM optimizes the IR while `-march=native` optimizes generated machine code
