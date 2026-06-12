# Plot Polish: Legends and Intermediate Axis Ticks

**Issue**: [#103](https://github.com/ahrefs/ocannl/issues/103)
**Milestone**: v1.0

## Status update (2026-06-12)

- Issue #103 is still **OPEN**, milestone v1.0 (GH milestone due date 2026-06-30; ROADMAP.md, the milestone authority, targets v1.0 for end of October 2026 under "Documentation, completeness, ergonomics").
- **No implementation has landed.** `tensor/PrintBox_utils.ml` is unchanged: the `plot` wrapper (now at line 80) still matches the snippet quoted below verbatim — no legend support, no labels channel, no intermediate ticks.
- The dependency pin is still `printbox-ext-plot >= 0.12` (dune-project) and 0.12 is the installed version; the upstream `plot_spec` type still carries no label field, so the "implement on top of `BPlot.box`" consequence still holds.
- All cited call-sites verified current: `test/training/moons_demo.ml` (three-spec plot), `test/operations/primitive_ops.ml`, `bin/compilation_speed.ml`, `test/operations/zero2hero_1of7*.ml` — none pass per-series identification.
- No repo-wide renames (basis/refines etc.) affect this proposal; it is presentation-layer only.
- Everything in Acceptance Criteria and Approach remains to do.

## Goal

OCANNL's text-based plots (built on `printbox-ext-plot`) currently support
multiple overlaid series but offer no way to tell them apart visually beyond
the glyph used in each cell. The accompanying axis labels show only the min
and max values of each axis, which makes it hard to read intermediate
positions.

Issue #103 asks for two related improvements:

1. **Legends** — "a legend box below the plot, right-aligned" so the reader
   can map glyphs to series.
2. **More numbers on axes** — at least a few intermediate tick labels in
   addition to the existing min/max.

Both are pure presentation polish. They unblock readable demos and tutorial
output for v1.0 (the moons demo, primitive-op visualizations, compilation
speed benchmarks).

## Acceptance Criteria

- [ ] **Legends:** when at least one plot spec is given an associated label,
      `PrintBox_utils.plot` renders a legend below the plot. The legend is a
      right-aligned box listing each labeled series as `<glyph> <label>`,
      where `<glyph>` is the spec's `content` box (or one cell of it). Plots
      with no labels render as before (no legend).
- [ ] **Axis ticks:** both axes display intermediate tick labels in addition
      to min/max. The number of intermediate labels is bounded (3–5
      defaults to a value that fits the configured plot size without visible
      collisions in the moons-demo and primitive-ops outputs). Tick values
      use the existing `concise_float` formatter at the configured precision.
- [ ] **Backward compatibility:** every existing call-site of
      `PrintBox_utils.plot` continues to compile and render without code
      changes; behavior of label-less plots is preserved (apart from the new
      intermediate ticks). Existing `.expected` snapshots are updated to
      reflect the new ticks.
- [ ] **Verifiable snapshot:** `moons_demo.expected` (or a small dedicated
      `legend_demo.expected` test) is updated to include the new legend and
      intermediate ticks for a multi-series plot — at minimum the moons
      scatterplot with the two `Scatterplot` series ("class A", "class B")
      and the `Boundary_map` ("decision boundary") labeled.
- [ ] **Single-series behavior:** legend is omitted when only one series has
      a label and the others have none, OR is rendered as a single-row box —
      either choice is acceptable provided it is consistent and documented.
- [ ] `dune runtest` (or its targeted equivalents for moons_demo and
      primitive_ops) passes after `.expected` files are regenerated.

## Context

### Current plotting code

`tensor/PrintBox_utils.ml` defines a single thin wrapper over
`printbox-ext-plot`:

```ocaml
let plot ?(as_canvas = false) ?x_label ?y_label ?axes ?size ?(small = false) specs =
  let default = BPlot.default_config in
  ...
  BPlot.box
    { BPlot.size; prec = Utils.settings.print_decimals_precision; axes; x_label; y_label; specs }
```

The `specs` argument is passed straight through to `BPlot.box` (i.e.,
`PrintBox_ext_plot.box`).

### What `printbox-ext-plot` 0.12 provides — and does not

The upstream `PrintBox_ext_plot.plot_spec` type
(`printbox-ext-plot/PrintBox_ext_plot.mli`) covers `Scatterplot`,
`Scatterbag`, `Line_plot`, `Boundary_map`, `Map`, `Line_plot_adaptive`. None
of the spec variants carry a label or color — every spec is identified
solely by the `content : PrintBox.t` glyph it stamps onto the canvas.

The `graph` record exposes `specs`, `x_label`, `y_label`, `size`, `axes`,
`prec` — no legend field, no per-tick configuration.

The internal `plot` function in `PrintBox_ext_plot.ml` (the rendering pass)
hard-codes axis labels as just `[maxy; miny]` on the y-axis and
`[minx; maxx]` on the x-axis using `concise_float`. There is no upstream
hook for additional ticks.

**Consequence:** both features have to be implemented in OCANNL on top of
`BPlot.box`. Either OCANNL composes its own outer frame (legend box, tick
labels) around the upstream-rendered canvas, or — for ticks — drops `axes:
true` and re-implements the axes layer entirely inside `PrintBox_utils`.
The proposed implementation lets the agent choose between these strategies
during planning; both are viable.

### Existing call-sites and what flows through them today

All call-sites pass `~x_label`/`~y_label` (or rely on defaults) but no
per-series identification. Sample sites:

- `test/training/moons_demo.ml` — three-spec plot: two `Scatterplot`s
  (glyphs `#` and `%`, conceptually class A and class B) plus a
  `Boundary_map` (glyphs `*` / `.`). This is the canonical "multi-series
  needs a legend" case.
- `test/operations/primitive_ops.ml` — three-spec plot with two
  `Scatterplot`s and a zero `Line_plot`; glyphs `*`, `#`, `-`.
- `bin/compilation_speed.ml` — single `Scatterplot` (glyph `#`).
- `test/operations/zero2hero_1of7*.ml` — analogous primitive-op plots.

No call-site currently has a series-name string that could automatically
seed legend labels — the API change must add a new optional channel for
labels.

### Backward compatibility constraint

All current call-sites pass `specs` as a `PrintBox_ext_plot.plot_spec list`.
Any change must let those calls compile unchanged. Two viable shapes (the
agent picks during planning):

- **Option A (wrapper specs):** introduce a `labeled_spec = { spec :
  plot_spec; label : string option }` and a parallel `~labeled_specs`
  argument. Existing callers keep using `specs : plot_spec list`.
- **Option B (parallel labels):** add a `?labels : string option list`
  argument that aligns 1:1 with `specs`. Length-mismatch is a programming
  error.

Either approach satisfies backward compat; option A is more explicit but
adds a new exported type.

### Snapshot update workflow

The repo uses `dune runtest` with `.expected` files (e.g.,
`moons_demo.expected`) for ASCII plot regression. Updating them is a normal
part of any rendering change; the agent should include the regenerated
files in the PR.

## Approach (suggested)

*Suggested approach — agents may deviate if they find a better path.*

1. Extend the `PrintBox_utils.plot` API with a way to associate a label
   with each spec (option B `~labels` is the smaller diff; option A
   `~labeled_specs` is cleaner — agent's call).
2. Build a legend `PrintBox.t` from the `(content, label)` pairs whose
   label is non-`None`. Render glyph cells using each spec's `content`
   directly so the visual matches what's plotted; pad/align labels into a
   right-aligned vlist.
3. For intermediate ticks: bypass the upstream `axes:true` path. Pass
   `axes:false` to `BPlot.box` (which omits the axes frame and gives just
   the canvas), then wrap the canvas in OCANNL's own grid that renders
   N intermediate ticks using `concise_float` at the configured precision.
   The min/max/span are recomputable from the same `specs` data the
   upstream uses (see `plot_canvas` in `PrintBox_ext_plot.ml` for the exact
   formulae). Defaults: 3 intermediate y-ticks, 3 intermediate x-ticks.
4. Compose: `vlist` of `[plot_with_axes; legend_box]` where `legend_box`
   is right-aligned via `align ~h:\`Right`.
5. Update `moons_demo.ml`, `primitive_ops.ml` (and similar call-sites) to
   pass labels for their multi-series plots. Single-series plots stay
   label-less.
6. Regenerate `.expected` files with `dune runtest --auto-promote` (or
   equivalent) and verify the diff visually matches the issue intent.

This is straightforward UI sugar. No design choice deserves a duo split.

## Scope

**In scope:**

- API extension on `PrintBox_utils.plot` for per-series labels.
- Legend rendering (right-aligned box below the plot).
- Intermediate axis tick labels with a sensible default count.
- Updating the moons demo and one or two other multi-series call-sites to
  pass labels, so the new feature is visible in regression output.
- `.expected` snapshot updates for the changed call-sites.

**Out of scope:**

- Color or styling beyond what `PrintBox` already supports (terminal output
  is character-cell; no ANSI color work is requested).
- HTML-specific legend tweaks beyond what falls out naturally (the upstream
  `html_handler` is unchanged because we wrap on top of `BPlot.box`).
- Configurable legend position (top/left/etc.); only the issue-requested
  bottom-right placement is required.
- Upstreaming legend support to `printbox-ext-plot`. Worth considering as
  a follow-up if the OCANNL implementation generalizes well, but not part
  of this task.

**Dependencies:** none. `printbox-ext-plot >= 0.12` is already pinned.
