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
  parameters; the `()` idiom makes that construction point explicit.
- `%cd` composition seams verified by ppx expansion (gh-465, `Train.sgd_one`/`grad_update`): an
  OCaml variable of `Asgns.comp` type in statement position splices verbatim (so a
  programmatically built fragment can sit inside a `%cd` body); an inline declaration `{ x }`
  hoists its let-binding to the top of the quotation, so declaring inside ONE `match` arm and
  referencing plain `x` from the other arms typechecks — but declaring the same name in two arms
  is a ppx-level "name clash" error. Gradients (`p.grad`) are readable only as DIRECT operands of
  an assignment: nested `(p.grad * s) + t` expands to `Option.map p.diff ... * s`, which does not
  typecheck — give the scaled read its own statement into an intermediate.

- A dimension capture needs a literal spec containing `=>`; this example captures the reduced axis.
  Doc-check `note_dimension_capture`: `let sum_last x = let open Ocannl.Nn_blocks.DSL_modules in [%op x ++ "ab => a" ["b"]]`.
