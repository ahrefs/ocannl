# Shapes & Einsum in OCANNL

{pause}

{#intro .definition title="Goal"}
Get productive with shapes, projections, and generalized einsum: from quick wins to advanced patterns like pooling and attention.

{pause up=intro}

- Work model: `batch | input -> output` rows per tensor, inferred lazily.
- Axis kinds aid expressivity; they don’t constrain semantics.
- Einsum specs align/permute/reduce axes declaratively.
- Operators `+*` and `++` map to `einsum` and `einsum1`.
- Shape logic shortcuts: `~logic:"."`, `~logic:"@"`, `~logic:"T"`.
-

---

{#axis-kinds .definition title="Axis Kinds = Expressivity"}
- Kinds: batch | input -> output; printed as `batch|input->output`.
- Semantics are not enforced; specs can pair any kinds.
- Why kinds help:
  - `~logic:"@"`: reduce all input axes of lhs1 with all output axes of lhs2 (tensor-matmul generalization).
  - Multiple row variables per tensor ⇒ patterns agnostic to number of axes (e.g., batch vs batch+microbatch; 1D vs 2D axial attention; separate head axis) without rewriting code.

{pause}

> Practical upshot: pick kinds to unlock concise specs and shape inference; don’t fear “misusing” them for nonstandard layouts.

---

{#notation .definition title="Notation vs NumPy"}
- OCANNL uses `->` inside one tensor to split input/output axes, so `=>` splits RHS→LHS across tensors.
- NumPy `"a,b->c"` becomes `"a;b=>c"` in OCANNL.
- Use `|` to separate batch axes: `"b|i->o"`.
- Multichar mode: any comma triggers it (can be trailing), e.g. `"i->o, => o->i"`.

{pause}

```text
# Single-char mode (no commas)
"ab;bc=>ac"

# Multichar mode (comma anywhere)
"row, col; col, feat => row, feat"
```

---

{#ellipsis .definition title="Row Variables & Ellipsis"}
- Ellipsis `"..."` or named `"..name.."` stand for a row variable (broadcastable block of axes).
- Lets you stay agnostic to the number and positions of axes; inference handles alignment.
- Dimension-1 axes broadcast as fixed index 0 in projections.

{pause}

```text
# Sum out everything → scalar
rhs ++ "...|... => 0"

# Keep batch/output, reduce inputs
rhs ++ "...|in1 in2 -> out1 out2 => ...|out1 out2"
```

---

{#ops .definition title="Operators Recap"}
- In tensor expressions (`%op`): `*` = tensor/matmul; `*.` = pointwise multiply.
- Einsum operators:
  - `t1 +* spec t2` → `Operation.einsum spec t1 t2` (multiply then sum-reduce where not preserved).
  - `t ++ spec` → `Operation.einsum1 spec t` (unary variant: permute/reduce/trace).
- In low-level `%cd`, choose accumulation with `=@^`, `=:+`, etc., and pass `~logic:"..."`.

{pause}

```ocaml
let%op y = a * b             (* tensor-matmul via kinds *)
let%op y = a +* "a;b=>c" b   (* explicit einsum *)
let%op z = x ++ "ab=>ba"     (* transpose *)
```

---

{#shape-flow .definition title="Shape Inference: Lifecycle"}
- Build-time: accumulate constraints; call `propagate_shapes`.
- JIT-time: `finish_inference` closes shapes (LUB, or 1/broadcastable) before codegen.
- Projections: freshen ids per op; solve unions; assign iterators for product dims; dim=1 → fixed idx 0.

{pause}

> Tip: rely on inference where possible; specify only what constrains semantics (e.g., reduction layout), not concrete sizes.

---

{#logic .definition title="Shape Logic Shortcuts"}
- `~logic:"."` pointwise broadcast (default for pointwise ops).
- `~logic:"@"` compose: match rhs2 output with rhs1 input (tensor-matmul).
- `~logic:"T"` transpose inputs/outputs.
- Any generalized einsum string works in `~logic:"..."` too.

{pause}

```ocaml
[%cd v =:+ t1 * t2 ~logic:"@"]   (* tensor-matmul by logic *)
[%cd v =:+ t  ~logic:"...=>..." ] (* general einsum from %cd *)
```

---

{#proj .definition title="Projections: What to Know"}
- Derived per assignment from the spec; unify equal projections via union-find.
- Non-product dims (already fixed) don’t get loops; others get iterators.
- Inequalities account for broadcasting; dim=1 → projection fixed to 0.

{pause}

> Debugging hint: if an index seems off, check which axes survive the LHS and which get reduced by your spec.

---

{#patterns .definition title="Common Patterns"}
- Batched matmul: `x +* "b|i->o; b|o->p => b|i->p" w`
- Outer sum: use `++` to expand and sum over matching axes.
- Trace/diagonal: `x ++ "ii=>"` (single-char) or `"row,row => 0"` (multi-char).

{pause}

```ocaml
let%op y = a +* "b|i->o; b|o->p => b|i->p" w
let%op tr = x ++ "ij=>0"      (* trace → scalar *)
```

---

{#conv .definition title="Convolution-Style Indexing"}
- Conv expression in specs: `stride * out + dilation * ker` inside an axis label.
- One input index per `(out, ker)` pair ⇒ great for window extraction.
- Trigger multichar with a comma anywhere in the spec.

{pause}

```ocaml
let stride = 2 and dilation = 1 in
(* Gather sliding windows with a unary einsum:
   windows[b, out, k] = x[b, stride*out + dilation*k] *)
let%op windows = x ++ "b| stride * out + dilation * k, => b| out, k,"
```

---

{#maxpool .definition title="Max Pooling via Einsum"}
Step 1: gather windows with a conv-style spec (previous slide).

{pause}

Step 2: reduce along kernel axis with max. Two options:

- Quick `%cd` reduce with custom accumulation:

```ocaml
let%op pooled =
  let windows = (* as above *) in
  [%cd y =:@^ windows ~logic:"b|out,k => b|out"]; y]
```

{pause}

- Or define your own op analogous to `einsum1` using max accumulation (user-extensible operations are straightforward):

```ocaml
(* Sketch: a max-reducing unary einsum variant *)
let%op max_reduce spec t =
  (* reduces axes dropped by LHS of [spec] using max (@^) *)
  [%cd v =:@^ t ~logic:spec]; v

(* Example: pool over kernel k, keep batch/out *)
let%op pooled = max_reduce "b|out,k => b|out" windows
```

Note: gradients for max pooling require subgradient/argmax logic; the forward shape usage is as above.

---

{#advanced .definition title="Advanced: Agnostic Layouts"}
- Use row variables and kinds to stay agnostic to axis counts/positions:
  - Batch vs batch+microbatch: `"...|i->o; ...|o->p => ...|i->p"`.
  - 1D vs 2D axial attention: keep `...` across spatial dims, isolate head axis explicitly.
- Prefer labels over sizes; inference propagates constraints and closes shapes before codegen.

{pause}

> Rule of thumb: encode relationships (who reduces with whom), not sizes.

---

{#pitfalls .remark title="Pitfalls & Tips"}
- Remember `|` where you intend distinct kinds; otherwise everything is output by default.
- Use a comma to switch to multichar mode when using long labels or conv expressions.
- `*` vs `*.`: tensor-multiply vs pointwise multiply in `%op`.
- Prefer `++`/`+*` for readability; drop to `%cd ~logic` when you need custom accumulators (max/min/and/or).
- Dimension-1 axes broadcast; relying on that can simplify specs dramatically.

{pause}

```ocaml
(* Tensor-multiply two blocks regardless of spatial ranks *)
let%op y = a * b                   (* via ~logic:"@" under the hood *)

(* Custom reduce: take min over channels *)
let%op min_over_c = [%cd z =:@- x ~logic:"...|c => ...|"]
```

---

{#wrapup .definition title="Where to Read Next"}
- Syntax deep dive: `docs/syntax_extensions.md` (einsum, `+*`, `++`, `%cd ~logic`).
- Shape/projection inference: `docs/shape_inference.md`, `lib/shape.mli`.
- Operations and semantics: selected parts of `lib/operation.ml`.
- Examples: `test/einsum/*`, `test/operations/*`, slides on backprop/codegen.

{pause up=wrapup}

Happy modeling! Design with labels, let inference do the rest.
