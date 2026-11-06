# Syntax extensions (%op / %cd)

What the two quotations mean, and where their scoping surprises.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Block-tensor delimiters map array `[|…|]` → batch, list `[…]` → output, tuple `(…)` → input
  axis; canonical nesting is array ⊃ list ⊃ tuple. Function-argument and einsum-operand tuples
  keep their OCaml meaning (distinct ppx arms), so `(a,b) ++^ …` is an operand pair, not a stack.
- `Einsum_parser.{binary,unary}_operators_with_generated_specs` and
  `concat_operator_with_generated_specs` own the unqualified operator spellings whose literal specs
  undergo runtime coefficient substitution. The PPX dispatch tables and source-policy inventory
  consume those same values; extend the shared set rather than copying a spelling into one side.
- `%op` inline-record init expressions (`{ w = kaiming normal1 () }`) are bound under the
  generated `open TDSL.O`, including when there is no unit parameter (gh-511). The no-unit-param
  form is not generative: `let%op f x = ...` closes over ONE shared param created at definition
  time. Use `let%op mk_f () x = ...` and apply `mk_f ()` when each model instance needs fresh
  parameters; the `()` idiom makes that construction point explicit. Einsum dimension captures
  (`[ "h"; "w" ]`) are NOT lifted with them: tagged `ocannl.capture` in `Ppx_shared`, they are
  bound by `bind_captures` at the top of the innermost enclosing function body, in `%op` and `%cd`
  alike, because a shared `delayed_var_ref` pins the first call's solution and a second size then
  fails inference (gh-ocannl-1507). A lifted initializer cannot name one; a capture that would
  shadow a lifted name is a clash, found through the `ocannl.bound_capture` marker `let_opt` checks.
- Label sources: only a toplevel `%op` function WITHOUT `()` reads its parameters' labels into
  the result's (the no-unit arm of `translate`); after `()`, no parameter is one. There an
  unlabelled parameter is read as `p.Tensor.value.Ir.Tnode.label`, so an unlabelled non-tensor is
  a type error at the parameter ("not a record type", or the mismatch at its first use), never a
  ppx message; optional ones are skipped (gh-ocannl-1787) since `?x` is an option and a default
  marks configuration; a labelled one is a source on `tensor_params`' evidence only -- a
  `Tensor.t` annotation, or an operand use with no rebinding (gh-ocannl-1885) -- so a
  forwarded-only labelled tensor contributes no label unless annotated, and `~hid_dim`/`~label`
  type-check. Before `()`, `find_label_param` takes `~label`, `?(label = ..)` (as `Some label`)
  and `?label` (the option itself) for the inline declarations' `?more_label`; before
  gh-ocannl-1885 an optional one was silently ignored, so `Demo_models.lenet`'s `w_logits` and
  `b_logits` lacked the `lenet` label. Without `()` the declarations are lifted to definition
  and cannot take it, so there the non-tensor parameter labelled `label` (any of the three forms)
  is itself a label source of the RESULT, at its position (gh-ocannl-1987; it was accepted and
  ignored); with `()` the result still gets the binding's name only -- an asymmetry by where each
  is possible, pinned by `test/ppx/test_label_param.ml`, which runs.
- Refusals that name the fix (gh-ocannl-1987, `test/ppx/test_param_errors.ml`, expansion only):
  `translate` threads `late`, the parameters bound after the lift point (every parameter of a
  no-unit toplevel function, and of any function below the toplevel one), and `guard_late`
  replaces an inline declaration's field or initializer naming one with an error node -- it was
  OCaml's "Unbound value", or a silent capture of an outer value of that name. A `let` in the
  body is NOT tracked (`let n = 3 in { b; o = [ n ] }` is still "Unbound value"). And an
  unlabelled, unannotated parameter with `non_tensor_evidence` (a stride/dilation identifier of
  a literal spec through `Ppx_shared.einsum_spec_identifiers`, the operand of `!.`/`!..`, the
  exponent of `**.`; never when the body rebinds the name) replaces the whole function with an
  error: without `()` always, after `()` only where the function gets a scheme, since an
  unschemed one (`[@@ocannl.no_scheme]`) takes an unlabelled int fine. Refusing rather than
  reclassifying keeps "unlabelled = tensor" a rule one can read code by, and can be relaxed
  later. The whole function is replaced because the type checker reports the first error it
  meets, and the mistyped use may come before an error node placed in the body. A change to what
  `%op` accepts also reaches the arm
  `tools/test-typecheck-scan-cases.sh` plants to be rejected through the ppx (its `#project` legs,
  CI's main shard 1), which `affected --harnesses` names for a change to any `tensor/ppx_*.ml`.
- Evaluation order (gh-ocannl-1869): where `%op` gives a literal a meaning of its own -- a block
  tensor's components, a concat's operand tuple, an einsum's capture list -- the expansion runs
  the members in source order (`Ppx_shared.in_source_order` let-binds each one that is not a
  variable or a constant, but the last; `collect_capture_labels` keeps the written order), so
  ids and capture uids ascend left to right (`test/ppx/test_eval_order.ml`, both compiled and
  run). Where it keeps OCaml's meaning the order is OCaml's, right to left: the operands of an
  operator or function application (`f x + g y` creates `g y` first, as `TDSL.O.(f x + g y)`
  does outside the ppx) and an argument tuple `f (a, b)`. That is the maintainer's decision
  (gh-ocannl-1986), not a gap to close: no value depends on it, and wrapping the operator and
  einsum arms in `in_source_order` would shift ids across most goldens. The uncurried primitive
  forms `op (a, b)` / `op (a, b, c)` expand to the curried call with the operator alone as head
  (gh-ocannl-1986; they used the whole application as the head and never type-checked), so their
  operands are created right to left too (`test/ppx/test_ppx_op.ml`, `_uncurried_*`). Inline
  declarations are not affected by either: they are lifted and bound in source order ahead of the
  expression. A new builder that splices several effectful expressions into one literal or
  argument list owes `in_source_order`.
- `%cd` composition seams verified by ppx expansion (gh-465, `Train.sgd_one`/`grad_update`): an
  OCaml variable of `Asgns.comp` type in statement position splices verbatim (so a
  programmatically built fragment can sit inside a `%cd` body); an inline declaration `{ x }`
  hoists its let-binding to the top of the quotation, so declaring inside ONE `match` arm and
  referencing plain `x` from the other arms typechecks — but declaring the same name in two arms
  is a ppx-level "name clash" error. Gradients (`p.grad`) are readable only as DIRECT operands of
  an assignment: nested `(p.grad * s) + t` expands to `Option.map p.diff ... * s`, which does not
  typecheck — give the scaled read its own statement into an intermediate.

- Automatic shape schemes (gh-ocannl-1751, `Tensor.Scheme.For_ppx`): `Ppx_op` marks the function a
  `()` stage returns (or a unit body's tail `fun`) with an internal `ocannl.scheme` attribute
  before translating, and the non-toplevel function arm wraps it; the no-unit toplevel arm wraps
  with `deferred`. Parameter classification is syntactic (`tensor_params`): unlabelled variables
  by convention, labelled ones only with direct-operand evidence. The trace adapter ignores the
  result, so results are never classified. Two traps: a body that builds a closure whose `{ w }`
  takes the default random initialization makes the enclosing trace refused (an RNG lazy in the
  window), so the enclosing function degrades to unschemed (with a stderr warning, once per
  function and reason, gh-ocannl-1791 -- an inline `%expect` block captures fd 2, so the warning
  lands in its expectation); and a body's OCaml side effects run once more per
  closure, so a test counting body runs opts out with `[@@ocannl.no_scheme]`.
- Option-valued tensors in schemes (gh-ocannl-1792): only an annotation classifies them
  (`~(p : Tensor.t option)`, `?(p : Tensor.t option)`; `?(p : Tensor.t)` is an OCaml type error), and
  `(x : Tensor.t)` is evidence for a forwarded labelled tensor. The handle keeps one slot per
  presence pattern, `[_variants]` entry points, all-required expansions byte-identical. The stacks
  scheme like the blocks: a checked application's body skips its nested applications, so a stack is
  one check (gh-ocannl-1814, shape-inference.md).
  A caller's `Some !@step_n` that nothing reads (dropout rate 0, batch norm) is an unconsumed forward
  root and shifts later printed ids by one.
- A dimension capture needs a literal spec containing `=>`; this example captures the reduced axis.
  Doc-check `note_dimension_capture`: `let sum_last x = let open Ocannl.Nn_blocks.DSL_modules in [%op x ++ "ab => a" ["b"]]`.
