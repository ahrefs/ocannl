# Local let-bindings in `%cd`

Issue: [#80](https://github.com/ahrefs/ocannl/issues/80)
ROADMAP: v1.0 ergonomics

## Goal

Allow ordinary non-recursive `let` bindings inside `%cd` so users can name
intermediate expressions without changing the generated assignment.

## Current state

`tensor/ppx_cd.ml` still rejects `Pexp_let`. Translation results already carry
the information a lexical binding needs: generated expression, expression
kind, projection slot, lifted value bindings, and optional array view.

The old proposal promised every OCaml let form, including `let rec`, and
specified a large matrix of implementation-level tests. Recursive local DSL
values have no demonstrated use and require cyclic type/slot inference; they
should remain rejected.

## Design

Thread a lexical environment through the translator. For each simple variable
binding:

1. translate all RHS expressions in the outer environment (preserving
   parallel `let ... and ...` scoping);
2. bind each generated RHS expression once;
3. translate the body in an environment mapping the source name to that
   generated identifier plus the RHS kind/slot metadata.

Environment lookup takes precedence over the name-based slot heuristic.
Nested bindings and shadowing therefore follow OCaml lexical rules. `_`
discards the value. Reject recursive bindings and nontrivial patterns with a
targeted PPX error until a real use case defines their semantics. Reserved DSL
metavariables such as `lhs`, `rhs1`, `v1`, and `t1` remain reserved and cannot
be shadowed.

## Completion criteria

- Simple, nested, shadowing, parallel, and discard bindings compile.
- Introducing a let produces the same `Assignments.comp` as the inlined form,
  including projection slots and embedded/lifted nodes.
- Array, tensor/value, scalar, and code-valued bindings retain their kind; an
  invalid use reports the bound name and expected context.
- RHS expressions are evaluated/generated once even when referenced more than
  once.
- `let rec`, unsupported patterns, and reserved-name shadowing fail clearly.
- Existing PPX expansion tests and the full `%cd` corpus pass.

Keep this as a translator feature. It should not introduce a runtime
assignment-level `Let` node unless sharing measurements later show that AST
binding is insufficient.
