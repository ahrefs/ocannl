# Compiling documentation examples

`dune build @test/operations/runtest-doc_examples` extracts selected Markdown examples and
compiles them against OCANNL and its actual `%op`/`%cd` PPX. The alias is part of `runtest`
and `@test/operations/scans`; `@check` also compiles the generated library. No example runs,
so backend constructors can be typechecked without needing their devices. This uses the
existing dependencies, rather than adding a separate Markdown runtime. Documentation edits
run the existing CI workflow because the checked snippets are compiler inputs.

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

## Excerpts of library code

Compilation cannot tell an excerpt from a paraphrase: a slide that restates a library function
compiles, or stays unchecked, however far it drifts from the code it claims to show. So a fence of
any language can declare where its code comes from, right after the language word:
`ocaml doc-excerpt=lib/nn_blocks.ml` says the block is a verbatim excerpt of that repository
file, and the extraction fails unless the block, with every whitespace run read as one space, is
a substring of the file. Indentation and line breaks may differ; nothing else may, comments
included, so commentary goes in the prose around the fence. `ocaml doc-adapted=lib/train.ml`
says the block is a simplification of that file; it is held only to the file existing, and the
prose should say "simplified". Either can precede `doc-check=` or `doc-skip`.

A fence captioned as code from an OCaml file -- the line directly above its opening says
`from <file>.ml` or `.mli`, such as a slide title "Real Examples from nn_blocks.ml" or
"**OCANNL** (from train.ml):" -- must carry one of the two declarations. A caption separated from
the fence by a blank line, or citing another language, is prose and is not read. Both slides the
shapes deck quotes from `lib/nn_blocks.ml` are checked excerpts; `doc_examples.expected` lists
each declaration beside the compiled groups.

The generated `_build/default/test/operations/doc_examples.inventory` lists every recognized
OCaml fence and selected agent-note span, including unchecked examples, with source locations
and reasons. `doc_examples.expected` pins every selected block
(by document and group) and explicit exclusions so removing
coverage needs a reviewed golden change. The corpus is all Markdown under `docs/`, plus the
root README and AGENTS guide, derived from a clean Dune source inventory. Symlinked proposal
copies appear under each source path.

The reader deliberately supports only top-level backtick or tilde fences with at most three
leading spaces, the same fences inside one level of block quote (`> ```ocaml`, as the slides'
containers write them; a quoted annotated fence that the quote leaves before its closing fence
fails), and the one-line agent-note form. An annotated unsupported form fails rather than
disappearing. Other languages (beyond the provenance declarations above), arbitrary inline API
mentions, indentation-only blocks, list-relative fences and nested block quotes are outside this
compilation contract. The inventory does not claim those are checked.

Compilation checks name resolution, types and PPX expansion. It cannot prove a simplified
optimizer has the same semantics as the real one, or that a valid memory gauge measures the
quantity a paragraph recommends. Behavioral claims still need pointers to implementation
or executed tests/transcripts; the training note cites `sgd_variants.ml` and
`data_parallel.ml` for optimizer behavior. This check complements those tests.
