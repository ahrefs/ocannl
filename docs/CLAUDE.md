# Writing Slipshow Presentations

Slipshow is a text-based presentation engine that compiles Markdown to interactive HTML presentations.

## Basic Syntax

- **Markdown base**: Uses CommonMark with extensions (tables, math, strikethrough)
- **Metadata**: Enclosed in `{...}` - can be standalone or attached to blocks/inlines
- **Pauses**: Use `{pause}` to create presentation steps
- **Blocks**: Use classes like `{.definition}`, `{.theorem}`, `{.example}`, `{.block}`, `{.alert}` with optional `title="..."`

## Key Features

### Navigation
- **Scrolling presentation**: Content flows like a papyrus, not discrete slides
- **Window movement**: `{up=id}`, `{down=id}`, `{center=id}` to control viewport
- **IDs**: Assign with `{#my-id}` to reference elements

### Grouping Content
- **Quote style**: Use `>` to group blocks together
- **Horizontal rules**: Use `---` to separate sections

### Interactive Elements
- **Drawing mode**: Press `w` to write, `h` to highlight, `e` to erase
- **Speaker view**: Press `s` for notes and timing
- **Custom scripts**: Use `{exec}` with `slip-script` code blocks

## Compilation

```bash
# Serve with auto-reload
slipshow serve presentation.md

# Compile once
slipshow compile presentation.md
```

## Example Structure

```markdown
# Title

Introduction paragraph.

{pause}

{.definition #important-def}
A **key concept** is defined here.

{pause up=important-def}

More content that scrolls to show the definition at top.
```

## Converting from Slides

When converting traditional slide presentations to slipshow:

- Use `{pause}` to break content into presentation steps
- Use `{pause up=id}` to scroll back to important context when needed
- Group related content logically rather than by slide boundaries
- Eliminate duplicate content by leveraging scrolling navigation

## Best Practices

- Use IDs strategically for navigation flow
- Leverage scrolling instead of creating duplicate content
- Group related content with `>` indentation
- Keep metadata readable by using referenced attributes for repetitive elements
- Structure content to flow naturally when read as text