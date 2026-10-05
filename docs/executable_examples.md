# Compiling documentation examples

`dune build @test/operations/runtest-doc_examples` extracts selected Markdown examples and
compiles them against OCANNL and its actual `%op`/`%cd` PPX. The alias is part of `runtest`
and `@test/operations/scans`; `@check` also compiles the generated library. No example runs,
so backend constructors can be typechecked without needing their devices. This uses the
existing dependencies, rather than adding a separate Markdown runtime.

Put `ocaml doc-check=group_name` on a fence's opening line. Each group within a document
becomes a separate module; blocks in the same group share bindings in document order. Different documents have
independent scopes, including symlinked copies. Write the required opens and function
arguments in the document itself: the extractor adds no hidden prelude. Compiler errors use
OCaml line directives to point back to the Markdown source.

Agent notes keep their flat prose dialect. They can select one complete OCaml code span on a
single line, spelled `Doc-check`, a backticked group name, a colon, and a backticked program
ending with a period. See the checked transform in
[virtualization-and-inlining.md](agent-notes/virtualization-and-inlining.md) and the capture in
[syntax-extensions.md](agent-notes/syntax-extensions.md). Longer programs belong in user docs
or tests, with a pointer from the note.

For an intentional exclusion, write `ocaml doc-skip` followed by a reason on the fence. Use
this for historical APIs or illustrative pseudocode; do not exclude a present-tense example
merely because it fails to compile. In proposals, dated historical sections may remain
exempt, but an update that recommends today's API must use a checked self-contained example.
The [routine-context proposal](proposals/centered-init-and-to-routine-context.md) demonstrates
both forms. Plain `ocaml` fences remain unchecked for gradual adoption.

The generated `_build/default/test/operations/doc_examples.inventory` lists every recognized
OCaml fence and selected agent-note span, including unchecked examples, with source locations
and reasons. `doc_examples.expected` pins every selected block
(by document and group) and explicit exclusions so removing
coverage needs a reviewed golden change. The corpus is all Markdown under `docs/`, plus the
root README and AGENTS guide, derived from a clean Dune source inventory. Symlinked proposal
copies appear under each source path.

The reader deliberately supports only top-level backtick or tilde fences with at most three
leading spaces and the one-line agent-note form. An annotated unsupported form fails rather
than disappearing. Other languages, arbitrary inline API mentions, indentation-only blocks,
list-relative fences and blockquoted fences are outside this compilation contract. The
inventory does not claim those are checked.

Compilation checks name resolution, types and PPX expansion. It cannot prove a simplified
optimizer has the same semantics as the real one, or that a valid memory gauge measures the
quantity a paragraph recommends. Behavioral claims still need pointers to implementation
or executed tests/transcripts; the training note cites `sgd_variants.ml` and
`data_parallel.ml` for optimizer behavior. This check complements those tests.
