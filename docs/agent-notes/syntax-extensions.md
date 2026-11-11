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
  They are derived from `Einsum_parser.{binary,unary}_operator_functions`/`concat_operator_function`,
  each operator paired with the named function it applies, and `named_spec_functions` (those
  functions plus `concat_sum` and the operator-less `einsum3`) is where the named forms' spellings
  live (gh-ocannl-2060): a named form's literal first argument is substituted by
  `Ppx_shared.substitute_named_form_spec`, which both PPXs apply to the applications they rebuild
  or pass through, only when the spec reads a runtime value (an identifier, or a plain `+` window's
  `use_padding`) -- otherwise the call is left as written. Its other arguments get the ordinary
  `%op` translation: a list or array literal there (`~capture_dims:[ d ]`, `concat`'s `[| x; y |]`)
  becomes a block tensor, so write it under `[%oc ...]`.
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
- Label sources: only a toplevel `%op` function WITHOUT `()` reads its tensor parameters' labels
  into the result's (the no-unit arm of `translate`); with `()`, no tensor parameter is one.
  Without `()`, an unlabelled parameter is read as `p.Tensor.value.Ir.Tnode.label`, so an unlabelled non-tensor is
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
  ignored); with `()` it also enriches the result after the binding's name (gh-ocannl-2059).
  An omitted `?label` adds nothing, a defaulted one adds its default, and inline declarations
  keep their labels. The result list is captured at the unit stage, under a name absent from the
  source, before a body binder or a parameter after `()` can shadow `label`. The result uses the
  same tensor-evidence exclusion as the no-unit arm; an explicitly ignored `~label:_` adds nothing.
  `test/ppx/test_label_param.ml` runs and pins these cases.
- Refusals that name the fix (gh-ocannl-1987, `test/ppx/test_param_errors.ml`, expansion only):
  `translate` threads `late`, the names bound after the lift point (every parameter of a
  no-unit toplevel function, and of any function below the toplevel one; and, gh-ocannl-2058,
  every binder of the body: `let` (a `let rec` in its right-hand sides too), `match` and
  `function` cases, `for` loops -- `late_binder`), and `guard_late` replaces an inline
  declaration's field or initializer naming one with an error node -- it was OCaml's "Unbound
  value", or a silent capture of an outer value of that name (a body `let n` shadowing a `~n`
  bound before `()` read the parameter). Every binder `translate` reaches is below the lift
  point: only the toplevel function's `()` is one, and `let%op f = let n = .. in fun () ..` makes
  that function non-toplevel. The `try` body and handlers, `lazy` body, constructor/variant
  arguments and `let*`/`and*` binding expressions and body are traversed too (gh-ocannl-2166).
  Handler patterns and all
  binding-operator patterns are late binders in their bodies; binding-operator right-hand sides
  keep the outer scope. Dimension captures bind inside the binding-operator callback, so an
  operator invoking it more than once gets fresh references for each invocation. Inline
  declarations still lift out of these wrappers, including `lazy`. Constructor/variant payloads
  preserve tuple grouping while translating their elements, including literals and inline records;
  use `[%oc ...]` for payloads that must keep their ordinary OCaml meaning. Binding-operator
  right-hand sides carry their pattern's label, like ordinary `let` bindings.
  An unlabelled, unannotated parameter with `non_tensor_evidence` (a stride/dilation identifier of
  a literal spec through `Ppx_shared.einsum_spec_identifiers` -- an operator's, or a named
  form's first positional argument (`Ppx_shared.named_form_spec`, gh-ocannl-2060) -- the operand
  of `!.`/`!..`, the exponent of `**.`, the argument of `Float.of_int`/`float_of_int`/
  `Int.to_float` spelled exactly; never when the body rebinds the name) replaces the whole function with an
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
- Application sites (gh-ocannl-2007): the generic application arm of `translate` (and the `|>`
  arm) wraps every application but a DSL operator's (`is_primitive_op`/`is_operator` heads, and
  the named einsum forms of `Einsum_parser.named_spec_functions`, which spell them) as
  `Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> app)`, which a scheme check inside it
  reads to name `<file>:<line>`. `__POS__` is filled in by the compiler at the application's
  location, so the `test/ppx/*_expected.ml` goldens spell no line. A contradiction while
  tracing a closure at its `()` application takes the same site from `Tensor.Scheme.trace_slot`
  through `trace_core` to `Shape.Scheme.trace` (gh-ocannl-2164); `check_definition` preserves it
  under its own context. Only contradictions name the site: fallback warnings and failed-rewind
  refusals keep their existing diagnostics. The
  `scheme_application_site` test pins both against the application's line. An arm that reads the
  applied function of a translated expression syntactically must undo the wrapper with `strip_at_site`:
  the inline-declaration initializer does (`add_module_qualifier_to_applied_function` qualifies
  its head with `NTDSL`; wrapped, `{ w = uniform () }` resolved `TDSL.O.uniform` and failed to
  type-check). Otherwise the thunk only types the application without its position's expected
  type and takes it out of tail position; no other `%op` code in the repository noticed.
- Type constraints in `%op` (gh-ocannl-2175) preserve translation of their expression, including
  result labels, application sites, inline declarations, scheme opt-outs and block-element context.
  Numeric-literal and tensor-container annotations keep their axis-basis meaning; `[%oc ...]`
  still opts out explicitly.
- `%cd` composition seams verified by ppx expansion (gh-465, `Train.sgd_one`/`grad_update`): an
  OCaml variable of `Asgns.comp` type in statement position splices verbatim (so a
  programmatically built fragment can sit inside a `%cd` body); an inline declaration `{ x }`
  hoists its let-binding to the top of the quotation, so declaring inside ONE `match` arm and
  referencing plain `x` from the other arms typechecks — but declaring the same name in two arms
  is a ppx-level "name clash" error. Gradients (`p.grad`) are readable only as DIRECT operands of
  an assignment: nested `(p.grad * s) + t` expands to `Option.map p.diff ... * s`, which does not
  typecheck — give the scaled read its own statement into an intermediate. A hand-written step
  meant to last wraps its `%cd` comp in `Asgns.persisting [ p.value ]` (gh-ocannl-2027; no syntax
  of its own); `Context.compile` refuses a routine re-establishing a node it marks
  (`Asgns.persists_undone`).

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

- Literal einsum operator captures are checked before substitution (gh-ocannl-2163):
  `collect_capture_labels` shares the check between `%op` and `%cd`, names the user's spec at the
  capture literal, and offers multichar mode for every multi-letter label in a single-char spec,
  even if some letters are absent (unlike Shape's conservative runtime hint).
  Rows, affine over/kernel labels, concat components and result-only labels are capturable;
  coefficient identifiers and fixed indices are not. `test/ppx/test_capture_errors.ml` pins the
  errors and accepted expansions; direct operation calls retain Shape's runtime fallback.

- Constant-only spec rendering keeps the parsed source if normalization would erase its multichar
  trigger: `"ab, => ab,"` must not reach Shape as `"ab => ab"`. Dynamic coefficient or padding
  substitutions retain an affine marker. `capture_multichar_hint` closes the documented repair's
  shape and checks both its captured dimension and rank (gh-ocannl-2163).
