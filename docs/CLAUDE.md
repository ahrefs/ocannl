# Writing Slipshow Presentations

Slipshow is a text-based presentation engine that compiles Markdown to interactive HTML presentations.

## Basic Syntax

- **Markdown base**: Uses CommonMark with extensions (tables, math, strikethrough)
- **Metadata**: Enclosed in `{...}` - can be standalone or attached to blocks/inlines
  - Metadata of the same block can be combined e.g. `{pause up}` instead of `{pause} {up}`
- **Pauses**: Use `{pause}` to create presentation steps
  - `{pause}` does not advance the presentation, use navigation to prevent slide overflow
- **Blocks**: Use classes like `{.definition}`, `{.theorem}`, `{.example}`, `{.block}`, `{.remark}` with optional `title="..."`

## Key Features

### Navigation
- **Scrolling presentation**: Content flows like a papyrus, not discrete slides
- **Window movement**: `{up=id}`, `{down=id}`, `{center=id}` to control viewport
  - `{up=id}` puts the element at the **top** of screen, revealing content **below** it
  - `{down=id}` puts the element at the **bottom** of screen, revealing content **above** it
  - `{center=id}` centers the element on screen
  - Must be used with pause e.g. `{pause down=id}` or with duration e.g. `{center="~duration:3 id"}` to take effect
- **IDs**: Assign with `{#my-id}` to reference elements

### Grouping Content
- **Quote style**: Use `>` to group blocks together
- **Horizontal rules**: Use `---` to separate sections

Standalone navigation action can be combined witth the following block metadata:

```markdown
{pause up=block-id}

{#block-id}
```

is equivalent to:

```markdown
{pause up #block-id}
```

and the latter is preferred.

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
- Use `{pause up=id}` to reveal content below an element (element at top)
- Use `{pause down=id}` to reveal content above an element (element at bottom)  
- Use `{pause center=id}` to focus attention on a specific element
- Group related content logically rather than by slide boundaries
- Eliminate duplicate content by leveraging scrolling navigation

## Best Practices

- Use IDs strategically for navigation flow
- Leverage scrolling instead of creating duplicate content
- Group related content with `>` indentation
- Keep metadata readable by using referenced attributes for repetitive elements
- Structure content to flow naturally when read as text